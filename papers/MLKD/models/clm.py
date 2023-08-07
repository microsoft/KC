# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from __future__ import barry_as_FLUFL
import json, math, os, torch
from utils.ood_metrics import compute_all_scores

from transformers import(
    AdamW,
    get_scheduler,
    get_constant_schedule
)
import joblib

def run_CLM(logger, args, model, data_loader, tokenized_datasets, tokenizer):
    if args.only_eval:
        train_id_scores, train_id_hiddens = evaluate_causal_LM(logger, model, data_loader['train'], args.batch_size_eval, len(tokenized_datasets['train']), output_dir=None)
    elif args.do_train:
        train_causal_LM(logger, args, model, data_loader, tokenized_datasets, tokenizer)
    
    for test_dataset in data_loader['test']:
        logger.info('Compute perplexity of validation (ID) data...')
        id_scores, id_hiddens = evaluate_causal_LM(logger, model, data_loader['validation'], args.batch_size_eval, len(tokenized_datasets['validation']), output_dir=None)
        
        logger.info('Compute perplexity of data (OOD) data...')
        ood_scores, od_hiddens = evaluate_causal_LM(logger, model, test_dataset, args.batch_size_eval, len(test_dataset), output_dir=None)
        
        # joblib.dump([id_scores, ood_scores], f"{args.output_dir}/scores_test_ood.pkl")
        
        res = compute_all_scores(id_scores=id_scores, ood_scores=ood_scores, output_dir=args.output_dir)
        keys = '\t\t'.join(res.keys())
        values = ','.join([str(i) for i in res.values()])
        logger.info(f'{keys}')
        logger.info(f'{values}')


def train_causal_LM(logger, args, model, data_loader, tokenized_datasets, tokenizer):
    logger.info('Train causal language model...')

    ## Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

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

    logger.info(f"  Num examples = {len(tokenized_datasets['train'])}")
    # logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size_train}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    # logger.info(f"  Total optimization steps = {args.max_train_steps}")

    completed_steps = 0
    epoch = 0
    total_loss_pre = 10000
    # for epoch in range(args.num_train_epochs):
    while True:
        model.train()
        total_loss = 0

        for step, batch in enumerate(data_loader['train']):
            for k in batch.keys():
                batch[k] = batch[k].to(args.device)
            batch['labels'] = batch['input_ids']
            outputs = model(**batch, output_hidden_states=True) # output['hidden_states']: list(n_layers) of (bs x seq_len x 768)
            loss = outputs.loss

            total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps

            loss.backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(data_loader['train']) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1

                if args.logging_steps > 0 and completed_steps % args.logging_steps == 0:
                    logger.info(f"Epoch: {epoch}\t Step: {completed_steps}/{args.max_train_steps} Loss: {loss}")
            
            if completed_steps >= args.max_train_steps:
                break
        
        # evaluate_causal_LM(logger, model, data_loader['validation'], args.batch_size_eval, len(tokenized_datasets['validation']), output_dir=None)
        total_loss /= step
        logger.info(f"Epoch {epoch} finished. Total Loss: {total_loss:.6f}\t Previous Total loss: {total_loss_pre:.6f}")
        if total_loss > total_loss_pre:
            break
        else:
            total_loss_pre = total_loss

        epoch += 1
    # Save model & tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Finish training. Save model and tokenizer to {args.output_dir}")


def evaluate_causal_LM(logger, model, data_loader, batch_size_eval, num_eval_examples, output_dir=None):
    """
    Evaluate `model` on `data_loader`.
        output_dir:     if not None, save evaluation results to `output_dir`.
    """
    model.eval()
    losses, hiddens = [], []
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    logger.info(f"  Num examples = {num_eval_examples}")
    logger.info(f"  Instantaneous batch size per device = {batch_size_eval}")

    for _, batch in enumerate(data_loader):
        for k in batch.keys():
            batch[k] = batch[k].to(model.device)
        batch['labels'] = batch['input_ids']

        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True)
            # import ipdb;ipdb.set_trace()

            # Shift so that tokens < n predict n
            shift_logits = outputs.logits[..., :-1, :].contiguous() # batch_size x seq_len x vocab_size
            shift_labels = batch['input_ids'][..., 1:].contiguous() # batch_size x seq_len
            # Flatten the tokens
            active = (batch['attention_mask'] == 1).view(-1)[:-1]
            active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
            active_labels = shift_labels.view(-1)[active]
            # hidden = outputs.hidden_states[-1].mul(batch['attention_mask'].unsqueeze(-1)).mean(dim=1)
            # hiddens.append(hidden)

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
    
    return ppls, hiddens

