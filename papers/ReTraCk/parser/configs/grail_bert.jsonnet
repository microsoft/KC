{
	"random_seed": 42,
	"numpy_seed": 42,
	"pytorch_seed": 42,
	"dataset_reader": {
		"type": "grail",
		"world": "GrailQA",
		"lazy": false,
		"loading_limit": -1,
		"use_bert": true,
		"tokenizer": {
			"type": "pretrained_transformer",
			"model_name": "bert-base-uncased",
			"max_length": 512
		},
		"token_indexers": {
			"bert": {
				"type": "pretrained_transformer",
				"model_name": "bert-base-uncased",
				"max_length": 512
			}
		},
		"encode_method": "all_domain"
	},
	"model": {
		"type": "grail",
		"text_embedder": {
			"token_embedders": {
				"bert": {
					"type": "pretrained_transformer",
					"model_name": "bert-base-uncased",
					"max_length": 512
				}
			}
		},
		"action_embedding_dim": 768,
		"entity_embedding_dim": 768,
		"text_encoder": {
			"type": "lstm",
			"input_size": 768,
			"hidden_size": 300,
			"bidirectional": true,
			"num_layers": 1
		},
		"decoder_beam_size": 5,
		"decoder_node_size": 20,
		"training_beam_size": 1,
		"max_decoding_steps": 30,
		"input_attention": {
			"type": "dot_product"
		},
		"dropout_rate": 0.1,
		"maximum_negative_chunk": 5,
		"maximum_negative_cand": 200,
		"dynamic_negative_ratio": 1.0,
		"utterance_agg_method": "first",
		"entity_order_method": "shuffle",
		"use_bert": true,
		"use_schema_encoder": false,
		"use_feature_score": false,
		"use_linking_embedding": false,
		"use_entity_anchor": true,
		"use_schema_as_input": true,
		"use_attention_select": false
	},
	"data_loader": {
		"batch_sampler": {
			"type": "bucket",
			"batch_size": 3,
			"sorting_keys": [
				"utterance"
			]
		}
	},
	"trainer": {
		"num_epochs": 10,
		"cuda_device": 0,
		"num_gradient_accumulation_steps": 1,
		"validation_metric": "+avg_exact_match",
		"optimizer": {
			"type": "adam",
			"parameter_groups": [
				[
					[
						".*text_embedder.*"
					],
					{
						"lr": 1e-5
					}
				]
			],
			"lr": 1e-3
		}
	}
}