# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json, math, os, torch
from sched import scheduler
from multiprocessing import reduction
from utils.ood_metrics import compute_all_scores
import torch.nn as nn
import numpy as np
import joblib

from transformers import(
    AdamW,
    get_scheduler,
    get_constant_schedule
)

def run_KD_CLM(logger, args, teacher, student, data_loader, tokenized_datasets, tokenizer, ood_score_metric='loss', temperature: float = 1.0, use_mse_loss: str = "none"):
    """
        Inputs:
            ood_score_metric:   choice between ['lm', 'loss'].
    """
    if args.do_train:
        student = train_KD_CLM(logger, args, teacher, student, data_loader, tokenized_datasets, tokenizer, temperature=temperature, use_mse_loss=use_mse_loss)

    # LM based OOD score
    for test_dataset in data_loader['test']:
        logger.info('Compute perplexity of validation (ID) data...')
        id_scores = evaluate_KD_CLM(logger, student, data_loader['validation'], args.batch_size_eval, len(tokenized_datasets['validation']), output_dir=None)

        logger.info('Compute perplexity of test (OOD) data...')
        ood_scores = evaluate_KD_CLM(logger, student, test_dataset, args.batch_size_eval, len(test_dataset), output_dir=None)

        joblib.dump([id_scores, ood_scores], f"{args.output_dir}/scores_test_ood.pkl")

        res = compute_all_scores(id_scores=id_scores, ood_scores=ood_scores, output_dir=args.output_dir)
        keys = '\t\t'.join(res.keys())
        values = ','.join([str(i) for i in res.values()])
        # with open(f"{args.loop_type}.csv", "a") as f:
        #     f.write(f"{args.stu_id},{args.tea_id},{values}\n")
        logger.info(f'{keys}')
        logger.info(f'{values}')
    
    # loss based OOD score
    # logger.info('Compute loss of modeling for validation (ID) data...')
    # id_scores = evaluate_KD_CLM_Loss(logger, teacher, student, data_loader['validation'], args.batch_size_eval, len(tokenized_datasets['validation']), output_dir=None, args=args)

    # logger.info('Compute loss of modeling for test (OOD) data...')
    # ood_scores = evaluate_KD_CLM_Loss(logger, teacher, student, data_loader['test'], args.batch_size_eval, len(tokenized_datasets['test']), output_dir=None, args=args)

    # for id_sc, ood_sc in zip(id_scores, ood_scores):
    #     res = compute_all_scores(id_scores=id_sc, ood_scores=ood_sc)
    #     keys = '\t\t'.join(res.keys())
    #     values = ','.join([str(i) for i in res.values()])
    #     logger.info(f'{keys}')
    #     logger.info(f'{values}')


def train_KD_CLM(logger, args, teacher, student, data_loader, tokenized_datasets, tokenizer, temperature=1.0, use_mse_loss="none"):
    logger.info('Train KD-CLM ...')

    kd_linear_tea = nn.Linear(768, 768).to(args.device)
    kd_linear_stu = nn.Linear(768, 768).to(args.device)
    # layer_unc = nn.Linear(768, 1).to(args.device)
    # layer_unc = nn.Linear(7 * 2, 1).to(args.device)
    # layer_unc.weight.data.fill_(1.0)
    layer_unc = nn.Parameter(torch.ones(7 * 2)).to(args.device)
    layer_probs = nn.Parameter(torch.ones(7, 12)).to(args.device)

    student.kd_linear_stu = kd_linear_stu
    student.kd_linear_tea = kd_linear_tea
    student.layer_unc = layer_unc
    student.layer_probs = layer_probs

    ## Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    unc = ["layer_unc"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in unc)],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate,
        },
        {
            "params": [p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in unc)],
            "weight_decay": 0.0,
            "lr": args.learning_rate,
        },
        {
            "params": [p for n, p in student.named_parameters() if any(nd in n for nd in unc)],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters)

    ## Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(data_loader['train']) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # lr_scheduler = get_scheduler(
    #     name=args.lr_scheduler_type,
    #     optimizer=optimizer,
    #     num_warmup_steps=args.num_warmup_steps,
    #     num_training_steps=args.max_train_steps,
    # )
    lr_scheduler = get_constant_schedule(optimizer=optimizer)

    total_batch_size = args.batch_size_train * args.gradient_accumulation_steps

    logger.info(f"  Each Epochs = {len(data_loader['train'])}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Num examples = {len(tokenized_datasets['train'])}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size_train}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    completed_steps = 0
    layers_to_align = [3, 6, 9, 12]
    loss_fn_mse = torch.nn.MSELoss(reduction='none')
    # def soft_cross_entropy(predicts, targets):
    #     student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    #     targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    #     res = - targets_prob * student_likelihood # batch_size x seq_len x vocab_size
    #     return torch.mean(torch.sum(res, dim=-1))

    epoch = 0
    total_loss_pre, total_loss_logits_pre, total_loss_hidden_pre = 10000, 10000, 10000
    # layer_unc = nn.Sequential(
    #     nn.Linear(768, 768 // 2),
    #     nn.ReLU(),
    #     nn.Linear(768 // 2, 1)
    # ).to(args.device)
    # for epoch in range(args.num_train_epochs):
    while True:
        teacher.eval()
        student.train()
        total_loss, total_loss_logits, total_loss_hidden = 0, 0, 0

        for step, batch in enumerate(data_loader['train']):
            for k in batch.keys():
                batch[k] = batch[k].to(args.device)
            batch['labels'] = batch['input_ids']

            student_outputs = student(**batch, output_hidden_states=True, output_attentions=True)
            with torch.no_grad():
                teacher_outputs = teacher(**batch, output_hidden_states=True, output_attentions=True)
            
            # Compute prediction-layer distillation
            loss_logits = nn.functional.kl_div(
                input=nn.functional.log_softmax((student_outputs.logits + 1e-10) / temperature, dim=-1),
                target=nn.functional.softmax((teacher_outputs.logits + 1e-10) / temperature, dim=-1),
                reduction="none",
            )
            # 10 * soft_cross_entropy(student_outputs.logits / temperature, teacher_outputs.logits / temperature)
            # loss_logits = loss_fn_mse(student_outputs.logits, teacher_outputs.logits) # batch_size x seq_len x vocab_size
            loss_logits = torch.sum(loss_logits, dim=-1) # batch_size x seq_len
            
            if args.intermediate_mode == "middle":
                # Compute embedding-layer loss
                loss_hiddens = []
                # l, r = 6, 10
                for stu in range(3, 10):
                    layer_hiddens = []
                    for tea in range(1, 13):
                        if args.loop_type == "kl":
                            loss_hidden = nn.functional.kl_div(
                                input=nn.functional.log_softmax((student_outputs.hidden_states[stu] + 1e-10) / temperature, dim=-1),
                                target=nn.functional.softmax(teacher_outputs.hidden_states[tea], dim=-1),
                                reduction="none",
                            )
                            loss_hidden = torch.sum(loss_hidden, dim=-1) 
                        elif args.loop_type == "mse":
                            loss_hidden = loss_fn_mse(teacher_outputs.hidden_states[tea], student_outputs.hidden_states[stu]) # batch_size x seq_len x hidden_dim
                            loss_hidden = torch.mean(loss_hidden, dim=-1) # batch_size x seq_len
                        elif args.loop_type == "linear_tea_mse":
                            loss_hidden = loss_fn_mse(student.kd_linear_tea(teacher_outputs.hidden_states[tea]), student_outputs.hidden_states[stu]) # batch_size x seq_len x hidden_dim
                            loss_hidden = torch.mean(loss_hidden, dim=-1) # batch_size x seq_len
                        elif args.loop_type == "linear_stu_mse":
                            loss_hidden = loss_fn_mse(teacher_outputs.hidden_states[tea], student.kd_linear_stu(student_outputs.hidden_states[stu])) # batch_size x seq_len x hidden_dim
                            loss_hidden = torch.mean(loss_hidden, dim=-1) # batch_size x seq_len
                        elif args.loop_type == "linear_both_mse":
                            loss_hidden = loss_fn_mse(student.kd_linear_tea(teacher_outputs.hidden_states[tea]), student.kd_linear_stu(student_outputs.hidden_states[stu])) # batch_size x seq_len x hidden_dim
                            loss_hidden = torch.mean(loss_hidden, dim=-1) # batch_size x seq_len
                        layer_hiddens.append(loss_hidden)
                    layer_hiddens = torch.stack(layer_hiddens, dim=-1)
                    # layer_hiddens = layer_hiddens.sort(dim=-1)[0][...,:2]
                    # layer_probs = nn.functional.softmax(layer_hiddens, dim=-1)
                    # layer_hiddens = layer_hiddens * layer_probs[stu - 3]
                    loss_hiddens.append(layer_hiddens.sort(dim=-1)[0][...,:2])
                    # loss_hiddens.append(layer_hiddens.sum(dim=-1))
                    
                # stu = torch.stack(student_outputs.hidden_states, dim=-1)
                # tea = torch.stack(teacher_outputs.hidden_states, dim=-1)
                # # stu = nn.functional.softmax(stu, dim=-1)
                # # tea = nn.functional.softmax(tea, dim=-1)
                # stu = stu.transpose(-2, -1)
                # tea = tea.transpose(-2, -1)
                # gap = tea - stu
                # gap_abs = gap.abs()
                # base = tea
                # loss_hiddens = torch.stack(loss_hiddens, dim=-1)
                # # loss_hidden = (loss_hiddens * layer_unc).sum(dim=-1)
                # unc = layer_unc(base[...,3:10,:]).squeeze(-1)
                # unc = torch.sigmoid(unc)
                # # print(loss_hiddens.shape, unc.shape)
                # # part1 = (loss_hiddens * unc).sum(-1)
                # # part2 = - torch.log(unc).sum(-1) * 0.1
                # # print(loss_hiddens, part1.sum(-1).mean(), part2.sum(-1).mean())
                # loss_hidden = (loss_hiddens * unc).sum(-1) - torch.log(unc).sum(-1) * 0.1
                # loss_hidden = loss_hidden / 5

                loss_hiddens = torch.stack(loss_hiddens, dim=-1)
                B, T, L, X = loss_hiddens.shape
                # if np.random.rand() < 0.01:
                #     print(student.layer_unc.weight)
                loss_hiddens = loss_hiddens.reshape(B, T, -1)
                # loss_hidden = loss_hiddens.sum(dim=-1)
                # import ipdb; ipdb.set_trace()
                loss_hidden = (loss_hiddens * student.layer_unc).sum(dim=-1)
                
                # loss_hidden = None
                # for i in layers_to_align:
                #     # Compute hidden-states loss
                #     # tmp = loss_fn_mse(teacher_outputs.hidden_states[i], student_outputs.hidden_states[i])
                #     # loss_hidden += torch.mean(tmp, dim=-1)
                #     # Compute attention loss
                #     teacher_att, student_att = teacher_outputs.attentions[i-1], student_outputs.attentions[i-1]
                #     # student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(student.device), student_att)
                #     # teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(teacher.device), teacher_att)
                    
                #     tmp = nn.functional.kl_div(
                #         input=torch.log(student_att + 1e-10),
                #         target=teacher_att,
                #         reduction="none",
                #     )
                #     # tmp = loss_fn_mse(student_att, teacher_att)
                #     if loss_hidden is None:
                #         loss_hidden = torch.sum(tmp, dim=-1).sum(dim=-2)
                #     else:
                #         loss_hidden += torch.sum(tmp, dim=-1).sum(dim=-2)
            elif args.intermediate_mode == "last":
                loss_hiddens = []
                # l, r = 6, 10
                for stu in range(12,13):
                    layer_hiddens = []
                    for tea in range(1, 13):
                        if args.loop_type == "kl":
                            loss_hidden = nn.functional.kl_div(
                                input=nn.functional.log_softmax((student_outputs.hidden_states[stu] + 1e-10) / temperature, dim=-1),
                                target=nn.functional.softmax(teacher_outputs.hidden_states[tea], dim=-1),
                                reduction="none",
                            )
                            loss_hidden = torch.sum(loss_hidden, dim=-1) 
                        elif args.loop_type == "mse":
                            loss_hidden = loss_fn_mse(teacher_outputs.hidden_states[tea], student_outputs.hidden_states[stu]) # batch_size x seq_len x hidden_dim
                            loss_hidden = torch.mean(loss_hidden, dim=-1) # batch_size x seq_len
                        elif args.loop_type == "linear_tea_mse":
                            loss_hidden = loss_fn_mse(kd_linear_tea(teacher_outputs.hidden_states[tea]), student_outputs.hidden_states[stu]) # batch_size x seq_len x hidden_dim
                            loss_hidden = torch.mean(loss_hidden, dim=-1) # batch_size x seq_len
                        elif args.loop_type == "linear_stu_mse":
                            loss_hidden = loss_fn_mse(teacher_outputs.hidden_states[tea], kd_linear_stu(student_outputs.hidden_states[stu])) # batch_size x seq_len x hidden_dim
                            loss_hidden = torch.mean(loss_hidden, dim=-1) # batch_size x seq_len
                        elif args.loop_type == "linear_both_mse":
                            loss_hidden = loss_fn_mse(kd_linear_tea(teacher_outputs.hidden_states[tea]), kd_linear_stu(student_outputs.hidden_states[stu])) # batch_size x seq_len x hidden_dim
                            loss_hidden = torch.mean(loss_hidden, dim=-1) # batch_size x seq_len
                        layer_hiddens.append(loss_hidden)
                    layer_hiddens = torch.stack(layer_hiddens, dim=-1)
                    loss_hidden = (layer_hiddens * layer_probs[0]).sum(dim=-1)

            active = (batch['attention_mask'] == 1).view(-1)
            
            loss_logits = torch.mean(loss_logits.view(-1)[active])
            if args.intermediate_mode != "none":
                loss_hidden = torch.mean(loss_hidden.view(-1)[active])
            
            loss = loss_logits
            if use_mse_loss == "both":
                loss += student_outputs.loss
            elif use_mse_loss == "single":
                loss = student_outputs.loss
            if args.intermediate_mode != "none":
                loss = loss + loss_hidden
            else:
                loss_hidden = torch.zeros_like(loss)
            # loss = student_outputs.loss
            # loss = loss.view(-1)[active]
            # loss = torch.mean(loss)

            total_loss += loss.detach().float()
            total_loss_logits += loss_logits.detach().float()
            total_loss_hidden += loss_hidden.detach().float()
            loss = loss / args.gradient_accumulation_steps

            loss.backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(data_loader['train']) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1

                if args.logging_steps > 0 and completed_steps % args.logging_steps == 0:
                    logger.info(f"Epoch: {epoch}. Step: {completed_steps}.\t Loss: {loss:.6f} Logits loss: {loss_logits.item():.6f} Hiddent loss: {loss_hidden.item():.6f}")

        if completed_steps >= args.max_train_steps:
            break
        
        total_loss_logits /= step
        total_loss_hidden /= step
        total_loss /= step
        logger.info(f"Epoch {epoch} finished. Total Loss: {total_loss:.6f}\t Total logits loss: {total_loss_logits:.6f}\t Total hiddent loss: {total_loss_hidden:.6f}")
        logger.info(f"Previous epoch Information. Total Loss: {total_loss_pre:.6f}\t Total logits loss: {total_loss_logits_pre:.6f}\t Total hiddent loss: {total_loss_hidden_pre:.6f}")
        
        if total_loss_logits > total_loss_logits_pre and total_loss_hidden > total_loss_hidden_pre:
            pass
        else:
            total_loss_logits_pre = total_loss_logits
            total_loss_hidden_pre = total_loss_hidden
            total_loss_pre = total_loss

        epoch += 1
        # evaluate_causal_LM(logger, student, data_loader['validation'], args.batch_size_eval, len(tokenized_datasets['validation']), output_dir=None)

    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Finish training. Save student and tokenizer to {args.output_dir}")

    return student


def evaluate_KD_CLM(logger, model, data_loader, batch_size_eval, num_eval_examples, output_dir=None):
    """
    Evaluate `model` on `data_loader`.
        output_dir:     if not None, save evaluation results to `output_dir`.
    """
    model.eval()
    losses = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    logger.info(f"  Num examples = {num_eval_examples}")
    logger.info(f"  Instantaneous batch size per device = {batch_size_eval}")

    for _, batch in enumerate(data_loader):
        for k in batch.keys():
            batch[k] = batch[k].to(model.device)
        batch['labels'] = batch['input_ids']

        with torch.no_grad():
            outputs = model(**batch)

            # Shift so that tokens < n predict n
            shift_logits = outputs.logits[..., :-1, :].contiguous() # batch_size x seq_len x vocab_size
            shift_labels = batch['input_ids'][..., 1:].contiguous() # batch_size x seq_len
            # Flatten the tokens
            active = (batch['attention_mask'] == 1).view(-1)[:-1]
            active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
            active_labels = shift_labels.view(-1)[active]

            loss = loss_fct(active_logits, active_labels)
            losses.append(torch.mean(loss).data)

    
    if len(losses) != num_eval_examples:
        raise ValueError('n_losses and n_examples do not match!')
    
    # compute ppls as OOD scores
    ppls = []
    for l in losses:
        try:
            p = math.exp(l)
        except OverflowError:
            p = float("inf")
        ppls.append(p)
    
    # compute language modeling perplexity
    try:
        perplexity = math.exp(torch.mean(torch.stack(losses)))
    except OverflowError:
        perplexity = float("inf")
    logger.info(f"Perplexity of causal language modeling: {perplexity}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f'Saving evalutaion results to {output_dir}...')
        with open('results.json', 'w', encoding='utf-8') as fw:
            json.dump({"perplexity": perplexity}, fw)
    
    return ppls

def evaluate_KD_CLM_Loss(logger, teacher, student, data_loader, batch_size_eval, num_eval_examples, output_dir=None, args=None, temperature=1.0):
    """
    Evaluate `model` on `data_loader`.
        output_dir:     if not None, save evaluation results to `output_dir`.
    """
    losses = []
    losses_logits = []
    losses_hidden = []

    loss_fn_mse = torch.nn.MSELoss(reduction='none')
    # def soft_cross_entropy(predicts, targets):
    #     student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    #     targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    #     res = - targets_prob * student_likelihood # batch_size x seq_len x vocab_size
    #     return torch.sum(res, dim=-1) # batch_size x seq_len

    teacher.eval()
    student.eval()

    logger.info(f"  Num examples = {num_eval_examples}")
    logger.info(f"  Instantaneous batch size per device = {batch_size_eval}")

    layers_to_align = [3, 6, 9, 12]

    for _, batch in enumerate(data_loader):
        for k in batch.keys():
            batch[k] = batch[k].to(teacher.device)
        batch['labels'] = batch['input_ids']

        with torch.no_grad():
            student_outputs = student(**batch, output_hidden_states=True) #, output_attentions=True)
            teacher_outputs = teacher(**batch, output_hidden_states=True) #, output_attentions=True)
            
            # Compute prediction-layer distillation
            loss_logits = nn.functional.kl_div(
                input=nn.functional.log_softmax((student_outputs.logits + 1e-10) / temperature, dim=-1),
                target=nn.functional.softmax(teacher_outputs.logits, dim=-1),
                reduction="none",
            )
            # 10 * soft_cross_entropy(student_outputs.logits / temperature, teacher_outputs.logits / temperature)
            # loss_logits = loss_fn_mse(student_outputs.logits, teacher_outputs.logits) # batch_size x seq_len x vocab_size
            loss_logits = torch.sum(loss_logits, dim=-1) # batch_size x seq_len

            loss_hiddens = []
            for stu in range(3, 10):
                layer_hiddens = []
                for tea in range(1, 13):
                    if args.loop_type == "kl":
                        loss_hidden = nn.functional.kl_div(
                            input=nn.functional.log_softmax((student_outputs.hidden_states[stu] + 1e-10) / temperature, dim=-1),
                            target=nn.functional.softmax(teacher_outputs.hidden_states[tea], dim=-1),
                            reduction="none",
                        )
                        loss_hidden = torch.sum(loss_hidden, dim=-1) 
                    elif args.loop_type == "mse":
                        loss_hidden = loss_fn_mse(teacher_outputs.hidden_states[tea], student_outputs.hidden_states[stu]) # batch_size x seq_len x hidden_dim
                        loss_hidden = torch.mean(loss_hidden, dim=-1) # batch_size x seq_len
                    elif args.loop_type == "linear_tea_mse":
                        loss_hidden = loss_fn_mse(student.kd_linear_tea(teacher_outputs.hidden_states[tea]), student_outputs.hidden_states[stu]) # batch_size x seq_len x hidden_dim
                        loss_hidden = torch.mean(loss_hidden, dim=-1) # batch_size x seq_len
                    elif args.loop_type == "linear_stu_mse":
                        loss_hidden = loss_fn_mse(teacher_outputs.hidden_states[tea], student.kd_linear_stu(student_outputs.hidden_states[stu])) # batch_size x seq_len x hidden_dim
                        loss_hidden = torch.mean(loss_hidden, dim=-1) # batch_size x seq_len
                    elif args.loop_type == "linear_both_mse":
                        loss_hidden = loss_fn_mse(student.kd_linear_tea(teacher_outputs.hidden_states[tea]), student.kd_linear_stu(student_outputs.hidden_states[stu])) # batch_size x seq_len x hidden_dim
                        loss_hidden = torch.mean(loss_hidden, dim=-1) # batch_size x seq_len
                    layer_hiddens.append(loss_hidden)
                layer_hiddens = torch.stack(layer_hiddens, dim=-1)
                # layer_hiddens = layer_hiddens.sort(dim=-1)[0][...,:2]
                # layer_probs = nn.functional.softmax(layer_hiddens, dim=-1)
                # layer_hiddens = layer_hiddens * layer_probs[stu - 3]
                loss_hiddens.append(layer_hiddens.sort(dim=-1)[0][...,:2].mean(dim=-1))
                # loss_hiddens.append(layer_hiddens.sum(dim=-1))

            loss_hiddens = torch.stack(loss_hiddens, dim=-1)
            loss_hidden = (loss_hiddens * student.layer_unc).sum(dim=-1)
                
            active = (batch['attention_mask'] == 1).view(-1)
            loss_logits = torch.mean(loss_logits.view(-1)[active])
            loss_hidden = torch.mean(loss_hidden.view(-1)[active])
            loss = loss_logits + loss_hidden

        losses.append(loss.item())
        losses_logits.append(loss_logits.item())
        losses_hidden.append(loss_hidden.item())
    
    if len(losses) != num_eval_examples:
        raise ValueError('n_losses and n_examples do not match!')
    
    return losses, losses_logits, losses_hidden
