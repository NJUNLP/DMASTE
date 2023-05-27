{
  SpanModel: {
    local span_model = self,

    // Mapping from target task to the metric used to assess performance on that task.
    local validation_metrics = {
      'ner': '+MEAN__ner_f1',
      'relation': '+MEAN__relation_f1',
    },

    ////////////////////

    // REQUIRED VALUES. Must be set by child class.

    // Paths to train, dev, and test data.
    data_paths :: error 'Must override `data_paths`',

    // Weights on the losses of the model components (e.g. NER, relation, etc).
    loss_weights :: error 'Must override `loss_weights`',

    // Make early stopping decisions based on performance for this task.
    // Options are: `['ner', 'relation']`
    target_task :: error 'Must override `target_task`',

    // DEFAULT VALUES. May be set by child class..
    bert_model :: 'bert-base-cased',
    // If using a different BERT, this number may be different. It's up to the user to set the
    // appropriate value.
    max_wordpieces_per_sentence :: 512,
    max_span_width :: 8,
    cuda_device :: -1,

    ////////////////////

    // All remaining values can be overridden using the `:+` mechanism
    // described in `doc/config.md`
    random_seed: 13370,
    numpy_seed: 1337,
    pytorch_seed: 133,
    dataset_reader: {
      type: 'span_model',
      token_indexers: {
        bert: {
          type: 'pretrained_transformer_mismatched',
          model_name: span_model.bert_model,
          max_length: span_model.max_wordpieces_per_sentence
        },
      },
      max_span_width: span_model.max_span_width
    },
    train_data_path: span_model.data_paths.train,
    validation_data_path: span_model.data_paths.validation,
    test_data_path: span_model.data_paths.test,
    // If provided, use pre-defined vocabulary. Else compute on the fly.
    model: {
      type: 'span_model',
      embedder: {
        token_embedders: {
          bert: {
            type: 'pretrained_transformer_mismatched',
            model_name: span_model.bert_model,
            max_length: span_model.max_wordpieces_per_sentence
          },
        },
      },
      initializer: {  // Initializer for shared span representations.
        regexes:
          [['_span_width_embedding.weight', { type: 'xavier_normal' }]],
      },
      module_initializer: {  // Initializer for component module weights.
        regexes:
          [
            ['.*weight', { type: 'xavier_normal' }],
            ['.*weight_matrix', { type: 'xavier_normal' }],
          ],
      },
      loss_weights: span_model.loss_weights,
      feature_size: 20,
      max_span_width: span_model.max_span_width,
      target_task: span_model.target_task,
      feedforward_params: {
        num_layers: 2,
        hidden_dims: 150,
        dropout: 0.4,
      },
      modules: {
        ner: {},
        relation: {
          spans_per_word: 0.5,
        },
      },
    },
    data_loader: {
      sampler: {
        type: "random",
      }
    },
    trainer: {
      checkpointer: {
        num_serialized_models_to_keep: 3,
      },
      num_epochs: 50,
      grad_norm: 5.0,
      cuda_device: span_model.cuda_device,
      validation_metric: validation_metrics[span_model.target_task],
      optimizer: {
        type: 'adamw',
        lr: 1e-3,
        weight_decay: 0.0,
        parameter_groups: [
          [
            ['_embedder'],
            {
              lr: 5e-5,
              weight_decay: 0.01,
              finetune: true,
            },
          ],
        ],
      },
      learning_rate_scheduler: {
        type: 'slanted_triangular'
      }
    },
  },
}
