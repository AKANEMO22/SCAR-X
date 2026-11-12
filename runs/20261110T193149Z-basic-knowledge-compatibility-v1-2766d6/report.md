# Basic knowledge data/codec compatibility

No neuralqualityscores. Within256cap does not mean rawcheckpoint can emit newtokens. Exactsurface and token-equivalent numeric outputs differ. No codecmigration/fix implemented.

{
  "run": "runs/20261110T193149Z-basic-knowledge-compatibility-v1-2766d6",
  "decision": "DATASET_NOT_LEARNABLE_UNDER_CURRENT_CODEC",
  "source_pool": "data/candidates/20261110T191533Z-basic-knowledge-repair-v1-231806/verified-pool.json",
  "source_sha256": "13bb051467b27caa10bf8d16039d52fa830baf77a5e87bd63e3520d8b96ed062",
  "verified_rows": 320,
  "facts": 40,
  "split_counts": {
    "unknown_confirmation": 32,
    "unknown_development": 32,
    "train": 128,
    "development": 64,
    "heldout_confirmation": 64
  },
  "audit": {
    "rows": 320,
    "existing_public_vocabulary": 24,
    "required_answer_tokens": 33,
    "union_vocabulary_size": 55,
    "within_supported_256_cap": true,
    "out_of_existing_checkpoint_vocabulary": 320,
    "oov_row_ids": [
      "api_chat_route_p01",
      "api_chat_route_p02",
      "api_chat_route_p03",
      "api_chat_route_p04",
      "api_chat_route_p05",
      "api_chat_route_p06",
      "api_chat_route_p07",
      "api_chat_route_p08",
      "api_checkpoint_route_p01",
      "api_checkpoint_route_p02",
      "api_checkpoint_route_p03",
      "api_checkpoint_route_p04",
      "api_checkpoint_route_p05",
      "api_checkpoint_route_p06",
      "api_checkpoint_route_p07",
      "api_checkpoint_route_p08",
      "api_export_route_p01",
      "api_export_route_p02",
      "api_export_route_p03",
      "api_export_route_p04",
      "api_export_route_p05",
      "api_export_route_p06",
      "api_export_route_p07",
      "api_export_route_p08",
      "api_host_p01",
      "api_host_p02",
      "api_host_p03",
      "api_host_p04",
      "api_host_p05",
      "api_host_p06",
      "api_host_p07",
      "api_host_p08",
      "api_model_id_p01",
      "api_model_id_p02",
      "api_model_id_p03",
      "api_model_id_p04",
      "api_model_id_p05",
      "api_model_id_p06",
      "api_model_id_p07",
      "api_model_id_p08",
      "api_models_route_p01",
      "api_models_route_p02",
      "api_models_route_p03",
      "api_models_route_p04",
      "api_models_route_p05",
      "api_models_route_p06",
      "api_models_route_p07",
      "api_models_route_p08",
      "api_port_p01",
      "api_port_p02",
      "api_port_p03",
      "api_port_p04",
      "api_port_p05",
      "api_port_p06",
      "api_port_p07",
      "api_port_p08",
      "api_status_route_p01",
      "api_status_route_p02",
      "api_status_route_p03",
      "api_status_route_p04",
      "api_status_route_p05",
      "api_status_route_p06",
      "api_status_route_p07",
      "api_status_route_p08",
      "config_adaptation_decay_p01",
      "config_adaptation_decay_p02",
      "config_adaptation_decay_p03",
      "config_adaptation_decay_p04",
      "config_adaptation_decay_p05",
      "config_adaptation_decay_p06",
      "config_adaptation_decay_p07",
      "config_adaptation_decay_p08",
      "config_adaptation_strength_p01",
      "config_adaptation_strength_p02",
      "config_adaptation_strength_p03",
      "config_adaptation_strength_p04",
      "config_adaptation_strength_p05",
      "config_adaptation_strength_p06",
      "config_adaptation_strength_p07",
      "config_adaptation_strength_p08",
      "config_anchor_strength_p01",
      "config_anchor_strength_p02",
      "config_anchor_strength_p03",
      "config_anchor_strength_p04",
      "config_anchor_strength_p05",
      "config_anchor_strength_p06",
      "config_anchor_strength_p07",
      "config_anchor_strength_p08",
      "config_fan_in_p01",
      "config_fan_in_p02",
      "config_fan_in_p03",
      "config_fan_in_p04",
      "config_fan_in_p05",
      "config_fan_in_p06",
      "config_fan_in_p07",
      "config_fan_in_p08",
      "config_feature_decay_p01",
      "config_feature_decay_p02",
      "config_feature_decay_p03",
      "config_feature_decay_p04",
      "config_feature_decay_p05",
      "config_feature_decay_p06",
      "config_feature_decay_p07",
      "config_feature_decay_p08",
      "config_hidden_p01",
      "config_hidden_p02",
      "config_hidden_p03",
      "config_hidden_p04",
      "config_hidden_p05",
      "config_hidden_p06",
      "config_hidden_p07",
      "config_hidden_p08",
      "config_hidden_lr_p01",
      "config_hidden_lr_p02",
      "config_hidden_lr_p03",
      "config_hidden_lr_p04",
      "config_hidden_lr_p05",
      "config_hidden_lr_p06",
      "config_hidden_lr_p07",
      "config_hidden_lr_p08",
      "config_inputs_p01",
      "config_inputs_p02",
      "config_inputs_p03",
      "config_inputs_p04",
      "config_inputs_p05",
      "config_inputs_p06",
      "config_inputs_p07",
      "config_inputs_p08",
      "config_leak_p01",
      "config_leak_p02",
      "config_leak_p03",
      "config_leak_p04",
      "config_leak_p05",
      "config_leak_p06",
      "config_leak_p07",
      "config_leak_p08",
      "config_neuron_p01",
      "config_neuron_p02",
      "config_neuron_p03",
      "config_neuron_p04",
      "config_neuron_p05",
      "config_neuron_p06",
      "config_neuron_p07",
      "config_neuron_p08",
      "config_outputs_p01",
      "config_outputs_p02",
      "config_outputs_p03",
      "config_outputs_p04",
      "config_outputs_p05",
      "config_outputs_p06",
      "config_outputs_p07",
      "config_outputs_p08",
      "config_plastic_rows_p01",
      "config_plastic_rows_p02",
      "config_plastic_rows_p03",
      "config_plastic_rows_p04",
      "config_plastic_rows_p05",
      "config_plastic_rows_p06",
      "config_plastic_rows_p07",
      "config_plastic_rows_p08",
      "config_ram_limit_p01",
      "config_ram_limit_p02",
      "config_ram_limit_p03",
      "config_ram_limit_p04",
      "config_ram_limit_p05",
      "config_ram_limit_p06",
      "config_ram_limit_p07",
      "config_ram_limit_p08",
      "config_readout_lr_p01",
      "config_readout_lr_p02",
      "config_readout_lr_p03",
      "config_readout_lr_p04",
      "config_readout_lr_p05",
      "config_readout_lr_p06",
      "config_readout_lr_p07",
      "config_readout_lr_p08",
      "config_threshold_p01",
      "config_threshold_p02",
      "config_threshold_p03",
      "config_threshold_p04",
      "config_threshold_p05",
      "config_threshold_p06",
      "config_threshold_p07",
      "config_threshold_p08",
      "config_trace_decay_p01",
      "config_trace_decay_p02",
      "config_trace_decay_p03",
      "config_trace_decay_p04",
      "config_trace_decay_p05",
      "config_trace_decay_p06",
      "config_trace_decay_p07",
      "config_trace_decay_p08",
      "g1_adaptation_strength_p01",
      "g1_adaptation_strength_p02",
      "g1_adaptation_strength_p03",
      "g1_adaptation_strength_p04",
      "g1_adaptation_strength_p05",
      "g1_adaptation_strength_p06",
      "g1_adaptation_strength_p07",
      "g1_adaptation_strength_p08",
      "g1_anchor_strength_p01",
      "g1_anchor_strength_p02",
      "g1_anchor_strength_p03",
      "g1_anchor_strength_p04",
      "g1_anchor_strength_p05",
      "g1_anchor_strength_p06",
      "g1_anchor_strength_p07",
      "g1_anchor_strength_p08",
      "g1_answer_token_cap_p01",
      "g1_answer_token_cap_p02",
      "g1_answer_token_cap_p03",
      "g1_answer_token_cap_p04",
      "g1_answer_token_cap_p05",
      "g1_answer_token_cap_p06",
      "g1_answer_token_cap_p07",
      "g1_answer_token_cap_p08",
      "g1_context_size_p01",
      "g1_context_size_p02",
      "g1_context_size_p03",
      "g1_context_size_p04",
      "g1_context_size_p05",
      "g1_context_size_p06",
      "g1_context_size_p07",
      "g1_context_size_p08",
      "g1_fan_in_p01",
      "g1_fan_in_p02",
      "g1_fan_in_p03",
      "g1_fan_in_p04",
      "g1_fan_in_p05",
      "g1_fan_in_p06",
      "g1_fan_in_p07",
      "g1_fan_in_p08",
      "g1_feature_decay_p01",
      "g1_feature_decay_p02",
      "g1_feature_decay_p03",
      "g1_feature_decay_p04",
      "g1_feature_decay_p05",
      "g1_feature_decay_p06",
      "g1_feature_decay_p07",
      "g1_feature_decay_p08",
      "g1_hidden_p01",
      "g1_hidden_p02",
      "g1_hidden_p03",
      "g1_hidden_p04",
      "g1_hidden_p05",
      "g1_hidden_p06",
      "g1_hidden_p07",
      "g1_hidden_p08",
      "g1_hidden_lr_p01",
      "g1_hidden_lr_p02",
      "g1_hidden_lr_p03",
      "g1_hidden_lr_p04",
      "g1_hidden_lr_p05",
      "g1_hidden_lr_p06",
      "g1_hidden_lr_p07",
      "g1_hidden_lr_p08",
      "g1_leak_p01",
      "g1_leak_p02",
      "g1_leak_p03",
      "g1_leak_p04",
      "g1_leak_p05",
      "g1_leak_p06",
      "g1_leak_p07",
      "g1_leak_p08",
      "g1_max_neurons_p01",
      "g1_max_neurons_p02",
      "g1_max_neurons_p03",
      "g1_max_neurons_p04",
      "g1_max_neurons_p05",
      "g1_max_neurons_p06",
      "g1_max_neurons_p07",
      "g1_max_neurons_p08",
      "g1_plastic_rows_p01",
      "g1_plastic_rows_p02",
      "g1_plastic_rows_p03",
      "g1_plastic_rows_p04",
      "g1_plastic_rows_p05",
      "g1_plastic_rows_p06",
      "g1_plastic_rows_p07",
      "g1_plastic_rows_p08",
      "g1_readout_lr_p01",
      "g1_readout_lr_p02",
      "g1_readout_lr_p03",
      "g1_readout_lr_p04",
      "g1_readout_lr_p05",
      "g1_readout_lr_p06",
      "g1_readout_lr_p07",
      "g1_readout_lr_p08",
      "g1_settle_ticks_p01",
      "g1_settle_ticks_p02",
      "g1_settle_ticks_p03",
      "g1_settle_ticks_p04",
      "g1_settle_ticks_p05",
      "g1_settle_ticks_p06",
      "g1_settle_ticks_p07",
      "g1_settle_ticks_p08",
      "g1_threshold_p01",
      "g1_threshold_p02",
      "g1_threshold_p03",
      "g1_threshold_p04",
      "g1_threshold_p05",
      "g1_threshold_p06",
      "g1_threshold_p07",
      "g1_threshold_p08",
      "g1_vocabulary_cap_p01",
      "g1_vocabulary_cap_p02",
      "g1_vocabulary_cap_p03",
      "g1_vocabulary_cap_p04",
      "g1_vocabulary_cap_p05",
      "g1_vocabulary_cap_p06",
      "g1_vocabulary_cap_p07",
      "g1_vocabulary_cap_p08",
      "g1_weight_clip_p01",
      "g1_weight_clip_p02",
      "g1_weight_clip_p03",
      "g1_weight_clip_p04",
      "g1_weight_clip_p05",
      "g1_weight_clip_p06",
      "g1_weight_clip_p07",
      "g1_weight_clip_p08"
    ],
    "over_24_target_tokens": [],
    "surface_roundtrip_mismatches": [
      {
        "id": "config_adaptation_decay_p01",
        "canonical": "0.95.",
        "rendered": "0. 95.",
        "tokens": [
          "0",
          ".",
          "95",
          "."
        ]
      },
      {
        "id": "config_adaptation_decay_p02",
        "canonical": "0.95.",
        "rendered": "0. 95.",
        "tokens": [
          "0",
          ".",
          "95",
          "."
        ]
      },
      {
        "id": "config_adaptation_decay_p03",
        "canonical": "0.95.",
        "rendered": "0. 95.",
        "tokens": [
          "0",
          ".",
          "95",
          "."
        ]
      },
      {
        "id": "config_adaptation_decay_p04",
        "canonical": "0.95.",
        "rendered": "0. 95.",
        "tokens": [
          "0",
          ".",
          "95",
          "."
        ]
      },
      {
        "id": "config_adaptation_decay_p05",
        "canonical": "0.95.",
        "rendered": "0. 95.",
        "tokens": [
          "0",
          ".",
          "95",
          "."
        ]
      },
      {
        "id": "config_adaptation_decay_p06",
        "canonical": "0.95.",
        "rendered": "0. 95.",
        "tokens": [
          "0",
          ".",
          "95",
          "."
        ]
      },
      {
        "id": "config_adaptation_decay_p07",
        "canonical": "0.95.",
        "rendered": "0. 95.",
        "tokens": [
          "0",
          ".",
          "95",
          "."
        ]
      },
      {
        "id": "config_adaptation_decay_p08",
        "canonical": "0.95.",
        "rendered": "0. 95.",
        "tokens": [
          "0",
          ".",
          "95",
          "."
        ]
      },
      {
        "id": "config_adaptation_strength_p01",
        "canonical": "0.08.",
        "rendered": "0. 08.",
        "tokens": [
          "0",
          ".",
          "08",
          "."
        ]
      },
      {
        "id": "config_adaptation_strength_p02",
        "canonical": "0.08.",
        "rendered": "0. 08.",
        "tokens": [
          "0",
          ".",
          "08",
          "."
        ]
      },
      {
        "id": "config_adaptation_strength_p03",
        "canonical": "0.08.",
        "rendered": "0. 08.",
        "tokens": [
          "0",
          ".",
          "08",
          "."
        ]
      },
      {
        "id": "config_adaptation_strength_p04",
        "canonical": "0.08.",
        "rendered": "0. 08.",
        "tokens": [
          "0",
          ".",
          "08",
          "."
        ]
      },
      {
        "id": "config_adaptation_strength_p05",
        "canonical": "0.08.",
        "rendered": "0. 08.",
        "tokens": [
          "0",
          ".",
          "08",
          "."
        ]
      },
      {
        "id": "config_adaptation_strength_p06",
        "canonical": "0.08.",
        "rendered": "0. 08.",
        "tokens": [
          "0",
          ".",
          "08",
          "."
        ]
      },
      {
        "id": "config_adaptation_strength_p07",
        "canonical": "0.08.",
        "rendered": "0. 08.",
        "tokens": [
          "0",
          ".",
          "08",
          "."
        ]
      },
      {
        "id": "config_adaptation_strength_p08",
        "canonical": "0.08.",
        "rendered": "0. 08.",
        "tokens": [
          "0",
          ".",
          "08",
          "."
        ]
      },
      {
        "id": "config_anchor_strength_p01",
        "canonical": "0.0001.",
        "rendered": "0. 0001.",
        "tokens": [
          "0",
          ".",
          "0001",
          "."
        ]
      },
      {
        "id": "config_anchor_strength_p02",
        "canonical": "0.0001.",
        "rendered": "0. 0001.",
        "tokens": [
          "0",
          ".",
          "0001",
          "."
        ]
      },
      {
        "id": "config_anchor_strength_p03",
        "canonical": "0.0001.",
        "rendered": "0. 0001.",
        "tokens": [
          "0",
          ".",
          "0001",
          "."
        ]
      },
      {
        "id": "config_anchor_strength_p04",
        "canonical": "0.0001.",
        "rendered": "0. 0001.",
        "tokens": [
          "0",
          ".",
          "0001",
          "."
        ]
      },
      {
        "id": "config_anchor_strength_p05",
        "canonical": "0.0001.",
        "rendered": "0. 0001.",
        "tokens": [
          "0",
          ".",
          "0001",
          "."
        ]
      },
      {
        "id": "config_anchor_strength_p06",
        "canonical": "0.0001.",
        "rendered": "0. 0001.",
        "tokens": [
          "0",
          ".",
          "0001",
          "."
        ]
      },
      {
        "id": "config_anchor_strength_p07",
        "canonical": "0.0001.",
        "rendered": "0. 0001.",
        "tokens": [
          "0",
          ".",
          "0001",
          "."
        ]
      },
      {
        "id": "config_anchor_strength_p08",
        "canonical": "0.0001.",
        "rendered": "0. 0001.",
        "tokens": [
          "0",
          ".",
          "0001",
          "."
        ]
      },
      {
        "id": "config_feature_decay_p01",
        "canonical": "0.92.",
        "rendered": "0. 92.",
        "tokens": [
          "0",
          ".",
          "92",
          "."
        ]
      },
      {
        "id": "config_feature_decay_p02",
        "canonical": "0.92.",
        "rendered": "0. 92.",
        "tokens": [
          "0",
          ".",
          "92",
          "."
        ]
      },
      {
        "id": "config_feature_decay_p03",
        "canonical": "0.92.",
        "rendered": "0. 92.",
        "tokens": [
          "0",
          ".",
          "92",
          "."
        ]
      },
      {
        "id": "config_feature_decay_p04",
        "canonical": "0.92.",
        "rendered": "0. 92.",
        "tokens": [
          "0",
          ".",
          "92",
          "."
        ]
      },
      {
        "id": "config_feature_decay_p05",
        "canonical": "0.92.",
        "rendered": "0. 92.",
        "tokens": [
          "0",
          ".",
          "92",
          "."
        ]
      },
      {
        "id": "config_feature_decay_p06",
        "canonical": "0.92.",
        "rendered": "0. 92.",
        "tokens": [
          "0",
          ".",
          "92",
          "."
        ]
      },
      {
        "id": "config_feature_decay_p07",
        "canonical": "0.92.",
        "rendered": "0. 92.",
        "tokens": [
          "0",
          ".",
          "92",
          "."
        ]
      },
      {
        "id": "config_feature_decay_p08",
        "canonical": "0.92.",
        "rendered": "0. 92.",
        "tokens": [
          "0",
          ".",
          "92",
          "."
        ]
      },
      {
        "id": "config_hidden_lr_p01",
        "canonical": "0.002.",
        "rendered": "0. 002.",
        "tokens": [
          "0",
          ".",
          "002",
          "."
        ]
      },
      {
        "id": "config_hidden_lr_p02",
        "canonical": "0.002.",
        "rendered": "0. 002.",
        "tokens": [
          "0",
          ".",
          "002",
          "."
        ]
      },
      {
        "id": "config_hidden_lr_p03",
        "canonical": "0.002.",
        "rendered": "0. 002.",
        "tokens": [
          "0",
          ".",
          "002",
          "."
        ]
      },
      {
        "id": "config_hidden_lr_p04",
        "canonical": "0.002.",
        "rendered": "0. 002.",
        "tokens": [
          "0",
          ".",
          "002",
          "."
        ]
      },
      {
        "id": "config_hidden_lr_p05",
        "canonical": "0.002.",
        "rendered": "0. 002.",
        "tokens": [
          "0",
          ".",
          "002",
          "."
        ]
      },
      {
        "id": "config_hidden_lr_p06",
        "canonical": "0.002.",
        "rendered": "0. 002.",
        "tokens": [
          "0",
          ".",
          "002",
          "."
        ]
      },
      {
        "id": "config_hidden_lr_p07",
        "canonical": "0.002.",
        "rendered": "0. 002.",
        "tokens": [
          "0",
          ".",
          "002",
          "."
        ]
      },
      {
        "id": "config_hidden_lr_p08",
        "canonical": "0.002.",
        "rendered": "0. 002.",
        "tokens": [
          "0",
          ".",
          "002",
          "."
        ]
      },
      {
        "id": "config_leak_p01",
        "canonical": "0.85.",
        "rendered": "0. 85.",
        "tokens": [
          "0",
          ".",
          "85",
          "."
        ]
      },
      {
        "id": "config_leak_p02",
        "canonical": "0.85.",
        "rendered": "0. 85.",
        "tokens": [
          "0",
          ".",
          "85",
          "."
        ]
      },
      {
        "id": "config_leak_p03",
        "canonical": "0.85.",
        "rendered": "0. 85.",
        "tokens": [
          "0",
          ".",
          "85",
          "."
        ]
      },
      {
        "id": "config_leak_p04",
        "canonical": "0.85.",
        "rendered": "0. 85.",
        "tokens": [
          "0",
          ".",
          "85",
          "."
        ]
      },
      {
        "id": "config_leak_p05",
        "canonical": "0.85.",
        "rendered": "0. 85.",
        "tokens": [
          "0",
          ".",
          "85",
          "."
        ]
      },
      {
        "id": "config_leak_p06",
        "canonical": "0.85.",
        "rendered": "0. 85.",
        "tokens": [
          "0",
          ".",
          "85",
          "."
        ]
      },
      {
        "id": "config_leak_p07",
        "canonical": "0.85.",
        "rendered": "0. 85.",
        "tokens": [
          "0",
          ".",
          "85",
          "."
        ]
      },
      {
        "id": "config_leak_p08",
        "canonical": "0.85.",
        "rendered": "0. 85.",
        "tokens": [
          "0",
          ".",
          "85",
          "."
        ]
      },
      {
        "id": "config_readout_lr_p01",
        "canonical": "0.08.",
        "rendered": "0. 08.",
        "tokens": [
          "0",
          ".",
          "08",
          "."
        ]
      },
      {
        "id": "config_readout_lr_p02",
        "canonical": "0.08.",
        "rendered": "0. 08.",
        "tokens": [
          "0",
          ".",
          "08",
          "."
        ]
      },
      {
        "id": "config_readout_lr_p03",
        "canonical": "0.08.",
        "rendered": "0. 08.",
        "tokens": [
          "0",
          ".",
          "08",
          "."
        ]
      },
      {
        "id": "config_readout_lr_p04",
        "canonical": "0.08.",
        "rendered": "0. 08.",
        "tokens": [
          "0",
          ".",
          "08",
          "."
        ]
      },
      {
        "id": "config_readout_lr_p05",
        "canonical": "0.08.",
        "rendered": "0. 08.",
        "tokens": [
          "0",
          ".",
          "08",
          "."
        ]
      },
      {
        "id": "config_readout_lr_p06",
        "canonical": "0.08.",
        "rendered": "0. 08.",
        "tokens": [
          "0",
          ".",
          "08",
          "."
        ]
      },
      {
        "id": "config_readout_lr_p07",
        "canonical": "0.08.",
        "rendered": "0. 08.",
        "tokens": [
          "0",
          ".",
          "08",
          "."
        ]
      },
      {
        "id": "config_readout_lr_p08",
        "canonical": "0.08.",
        "rendered": "0. 08.",
        "tokens": [
          "0",
          ".",
          "08",
          "."
        ]
      },
      {
        "id": "config_threshold_p01",
        "canonical": "0.65.",
        "rendered": "0. 65.",
        "tokens": [
          "0",
          ".",
          "65",
          "."
        ]
      },
      {
        "id": "config_threshold_p02",
        "canonical": "0.65.",
        "rendered": "0. 65.",
        "tokens": [
          "0",
          ".",
          "65",
          "."
        ]
      },
      {
        "id": "config_threshold_p03",
        "canonical": "0.65.",
        "rendered": "0. 65.",
        "tokens": [
          "0",
          ".",
          "65",
          "."
        ]
      },
      {
        "id": "config_threshold_p04",
        "canonical": "0.65.",
        "rendered": "0. 65.",
        "tokens": [
          "0",
          ".",
          "65",
          "."
        ]
      },
      {
        "id": "config_threshold_p05",
        "canonical": "0.65.",
        "rendered": "0. 65.",
        "tokens": [
          "0",
          ".",
          "65",
          "."
        ]
      },
      {
        "id": "config_threshold_p06",
        "canonical": "0.65.",
        "rendered": "0. 65.",
        "tokens": [
          "0",
          ".",
          "65",
          "."
        ]
      },
      {
        "id": "config_threshold_p07",
        "canonical": "0.65.",
        "rendered": "0. 65.",
        "tokens": [
          "0",
          ".",
          "65",
          "."
        ]
      },
      {
        "id": "config_threshold_p08",
        "canonical": "0.65.",
        "rendered": "0. 65.",
        "tokens": [
          "0",
          ".",
          "65",
          "."
        ]
      },
      {
        "id": "config_trace_decay_p01",
        "canonical": "0.9.",
        "rendered": "0. 9.",
        "tokens": [
          "0",
          ".",
          "9",
          "."
        ]
      },
      {
        "id": "config_trace_decay_p02",
        "canonical": "0.9.",
        "rendered": "0. 9.",
        "tokens": [
          "0",
          ".",
          "9",
          "."
        ]
      },
      {
        "id": "config_trace_decay_p03",
        "canonical": "0.9.",
        "rendered": "0. 9.",
        "tokens": [
          "0",
          ".",
          "9",
          "."
        ]
      },
      {
        "id": "config_trace_decay_p04",
        "canonical": "0.9.",
        "rendered": "0. 9.",
        "tokens": [
          "0",
          ".",
          "9",
          "."
        ]
      },
      {
        "id": "config_trace_decay_p05",
        "canonical": "0.9.",
        "rendered": "0. 9.",
        "tokens": [
          "0",
          ".",
          "9",
          "."
        ]
      },
      {
        "id": "config_trace_decay_p06",
        "canonical": "0.9.",
        "rendered": "0. 9.",
        "tokens": [
          "0",
          ".",
          "9",
          "."
        ]
      },
      {
        "id": "config_trace_decay_p07",
        "canonical": "0.9.",
        "rendered": "0. 9.",
        "tokens": [
          "0",
          ".",
          "9",
          "."
        ]
      },
      {
        "id": "config_trace_decay_p08",
        "canonical": "0.9.",
        "rendered": "0. 9.",
        "tokens": [
          "0",
          ".",
          "9",
          "."
        ]
      },
      {
        "id": "g1_adaptation_strength_p01",
        "canonical": "0.02.",
        "rendered": "0. 02.",
        "tokens": [
          "0",
          ".",
          "02",
          "."
        ]
      },
      {
        "id": "g1_adaptation_strength_p02",
        "canonical": "0.02.",
        "rendered": "0. 02.",
        "tokens": [
          "0",
          ".",
          "02",
          "."
        ]
      },
      {
        "id": "g1_adaptation_strength_p03",
        "canonical": "0.02.",
        "rendered": "0. 02.",
        "tokens": [
          "0",
          ".",
          "02",
          "."
        ]
      },
      {
        "id": "g1_adaptation_strength_p04",
        "canonical": "0.02.",
        "rendered": "0. 02.",
        "tokens": [
          "0",
          ".",
          "02",
          "."
        ]
      },
      {
        "id": "g1_adaptation_strength_p05",
        "canonical": "0.02.",
        "rendered": "0. 02.",
        "tokens": [
          "0",
          ".",
          "02",
          "."
        ]
      },
      {
        "id": "g1_adaptation_strength_p06",
        "canonical": "0.02.",
        "rendered": "0. 02.",
        "tokens": [
          "0",
          ".",
          "02",
          "."
        ]
      },
      {
        "id": "g1_adaptation_strength_p07",
        "canonical": "0.02.",
        "rendered": "0. 02.",
        "tokens": [
          "0",
          ".",
          "02",
          "."
        ]
      },
      {
        "id": "g1_adaptation_strength_p08",
        "canonical": "0.02.",
        "rendered": "0. 02.",
        "tokens": [
          "0",
          ".",
          "02",
          "."
        ]
      },
      {
        "id": "g1_feature_decay_p01",
        "canonical": "0.25.",
        "rendered": "0. 25.",
        "tokens": [
          "0",
          ".",
          "25",
          "."
        ]
      },
      {
        "id": "g1_feature_decay_p02",
        "canonical": "0.25.",
        "rendered": "0. 25.",
        "tokens": [
          "0",
          ".",
          "25",
          "."
        ]
      },
      {
        "id": "g1_feature_decay_p03",
        "canonical": "0.25.",
        "rendered": "0. 25.",
        "tokens": [
          "0",
          ".",
          "25",
          "."
        ]
      },
      {
        "id": "g1_feature_decay_p04",
        "canonical": "0.25.",
        "rendered": "0. 25.",
        "tokens": [
          "0",
          ".",
          "25",
          "."
        ]
      },
      {
        "id": "g1_feature_decay_p05",
        "canonical": "0.25.",
        "rendered": "0. 25.",
        "tokens": [
          "0",
          ".",
          "25",
          "."
        ]
      },
      {
        "id": "g1_feature_decay_p06",
        "canonical": "0.25.",
        "rendered": "0. 25.",
        "tokens": [
          "0",
          ".",
          "25",
          "."
        ]
      },
      {
        "id": "g1_feature_decay_p07",
        "canonical": "0.25.",
        "rendered": "0. 25.",
        "tokens": [
          "0",
          ".",
          "25",
          "."
        ]
      },
      {
        "id": "g1_feature_decay_p08",
        "canonical": "0.25.",
        "rendered": "0. 25.",
        "tokens": [
          "0",
          ".",
          "25",
          "."
        ]
      },
      {
        "id": "g1_hidden_lr_p01",
        "canonical": "0.0001.",
        "rendered": "0. 0001.",
        "tokens": [
          "0",
          ".",
          "0001",
          "."
        ]
      },
      {
        "id": "g1_hidden_lr_p02",
        "canonical": "0.0001.",
        "rendered": "0. 0001.",
        "tokens": [
          "0",
          ".",
          "0001",
          "."
        ]
      },
      {
        "id": "g1_hidden_lr_p03",
        "canonical": "0.0001.",
        "rendered": "0. 0001.",
        "tokens": [
          "0",
          ".",
          "0001",
          "."
        ]
      },
      {
        "id": "g1_hidden_lr_p04",
        "canonical": "0.0001.",
        "rendered": "0. 0001.",
        "tokens": [
          "0",
          ".",
          "0001",
          "."
        ]
      },
      {
        "id": "g1_hidden_lr_p05",
        "canonical": "0.0001.",
        "rendered": "0. 0001.",
        "tokens": [
          "0",
          ".",
          "0001",
          "."
        ]
      },
      {
        "id": "g1_hidden_lr_p06",
        "canonical": "0.0001.",
        "rendered": "0. 0001.",
        "tokens": [
          "0",
          ".",
          "0001",
          "."
        ]
      },
      {
        "id": "g1_hidden_lr_p07",
        "canonical": "0.0001.",
        "rendered": "0. 0001.",
        "tokens": [
          "0",
          ".",
          "0001",
          "."
        ]
      },
      {
        "id": "g1_hidden_lr_p08",
        "canonical": "0.0001.",
        "rendered": "0. 0001.",
        "tokens": [
          "0",
          ".",
          "0001",
          "."
        ]
      },
      {
        "id": "g1_leak_p01",
        "canonical": "0.65.",
        "rendered": "0. 65.",
        "tokens": [
          "0",
          ".",
          "65",
          "."
        ]
      },
      {
        "id": "g1_leak_p02",
        "canonical": "0.65.",
        "rendered": "0. 65.",
        "tokens": [
          "0",
          ".",
          "65",
          "."
        ]
      },
      {
        "id": "g1_leak_p03",
        "canonical": "0.65.",
        "rendered": "0. 65.",
        "tokens": [
          "0",
          ".",
          "65",
          "."
        ]
      },
      {
        "id": "g1_leak_p04",
        "canonical": "0.65.",
        "rendered": "0. 65.",
        "tokens": [
          "0",
          ".",
          "65",
          "."
        ]
      },
      {
        "id": "g1_leak_p05",
        "canonical": "0.65.",
        "rendered": "0. 65.",
        "tokens": [
          "0",
          ".",
          "65",
          "."
        ]
      },
      {
        "id": "g1_leak_p06",
        "canonical": "0.65.",
        "rendered": "0. 65.",
        "tokens": [
          "0",
          ".",
          "65",
          "."
        ]
      },
      {
        "id": "g1_leak_p07",
        "canonical": "0.65.",
        "rendered": "0. 65.",
        "tokens": [
          "0",
          ".",
          "65",
          "."
        ]
      },
      {
        "id": "g1_leak_p08",
        "canonical": "0.65.",
        "rendered": "0. 65.",
        "tokens": [
          "0",
          ".",
          "65",
          "."
        ]
      },
      {
        "id": "g1_readout_lr_p01",
        "canonical": "0.025.",
        "rendered": "0. 025.",
        "tokens": [
          "0",
          ".",
          "025",
          "."
        ]
      },
      {
        "id": "g1_readout_lr_p02",
        "canonical": "0.025.",
        "rendered": "0. 025.",
        "tokens": [
          "0",
          ".",
          "025",
          "."
        ]
      },
      {
        "id": "g1_readout_lr_p03",
        "canonical": "0.025.",
        "rendered": "0. 025.",
        "tokens": [
          "0",
          ".",
          "025",
          "."
        ]
      },
      {
        "id": "g1_readout_lr_p04",
        "canonical": "0.025.",
        "rendered": "0. 025.",
        "tokens": [
          "0",
          ".",
          "025",
          "."
        ]
      },
      {
        "id": "g1_readout_lr_p05",
        "canonical": "0.025.",
        "rendered": "0. 025.",
        "tokens": [
          "0",
          ".",
          "025",
          "."
        ]
      },
      {
        "id": "g1_readout_lr_p06",
        "canonical": "0.025.",
        "rendered": "0. 025.",
        "tokens": [
          "0",
          ".",
          "025",
          "."
        ]
      },
      {
        "id": "g1_readout_lr_p07",
        "canonical": "0.025.",
        "rendered": "0. 025.",
        "tokens": [
          "0",
          ".",
          "025",
          "."
        ]
      },
      {
        "id": "g1_readout_lr_p08",
        "canonical": "0.025.",
        "rendered": "0. 025.",
        "tokens": [
          "0",
          ".",
          "025",
          "."
        ]
      },
      {
        "id": "g1_threshold_p01",
        "canonical": "0.5.",
        "rendered": "0. 5.",
        "tokens": [
          "0",
          ".",
          "5",
          "."
        ]
      },
      {
        "id": "g1_threshold_p02",
        "canonical": "0.5.",
        "rendered": "0. 5.",
        "tokens": [
          "0",
          ".",
          "5",
          "."
        ]
      },
      {
        "id": "g1_threshold_p03",
        "canonical": "0.5.",
        "rendered": "0. 5.",
        "tokens": [
          "0",
          ".",
          "5",
          "."
        ]
      },
      {
        "id": "g1_threshold_p04",
        "canonical": "0.5.",
        "rendered": "0. 5.",
        "tokens": [
          "0",
          ".",
          "5",
          "."
        ]
      },
      {
        "id": "g1_threshold_p05",
        "canonical": "0.5.",
        "rendered": "0. 5.",
        "tokens": [
          "0",
          ".",
          "5",
          "."
        ]
      },
      {
        "id": "g1_threshold_p06",
        "canonical": "0.5.",
        "rendered": "0. 5.",
        "tokens": [
          "0",
          ".",
          "5",
          "."
        ]
      },
      {
        "id": "g1_threshold_p07",
        "canonical": "0.5.",
        "rendered": "0. 5.",
        "tokens": [
          "0",
          ".",
          "5",
          "."
        ]
      },
      {
        "id": "g1_threshold_p08",
        "canonical": "0.5.",
        "rendered": "0. 5.",
        "tokens": [
          "0",
          ".",
          "5",
          "."
        ]
      },
      {
        "id": "g1_weight_clip_p01",
        "canonical": "20.0.",
        "rendered": "20. 0.",
        "tokens": [
          "20",
          ".",
          "0",
          "."
        ]
      },
      {
        "id": "g1_weight_clip_p02",
        "canonical": "20.0.",
        "rendered": "20. 0.",
        "tokens": [
          "20",
          ".",
          "0",
          "."
        ]
      },
      {
        "id": "g1_weight_clip_p03",
        "canonical": "20.0.",
        "rendered": "20. 0.",
        "tokens": [
          "20",
          ".",
          "0",
          "."
        ]
      },
      {
        "id": "g1_weight_clip_p04",
        "canonical": "20.0.",
        "rendered": "20. 0.",
        "tokens": [
          "20",
          ".",
          "0",
          "."
        ]
      },
      {
        "id": "g1_weight_clip_p05",
        "canonical": "20.0.",
        "rendered": "20. 0.",
        "tokens": [
          "20",
          ".",
          "0",
          "."
        ]
      },
      {
        "id": "g1_weight_clip_p06",
        "canonical": "20.0.",
        "rendered": "20. 0.",
        "tokens": [
          "20",
          ".",
          "0",
          "."
        ]
      },
      {
        "id": "g1_weight_clip_p07",
        "canonical": "20.0.",
        "rendered": "20. 0.",
        "tokens": [
          "20",
          ".",
          "0",
          "."
        ]
      },
      {
        "id": "g1_weight_clip_p08",
        "canonical": "20.0.",
        "rendered": "20. 0.",
        "tokens": [
          "20",
          ".",
          "0",
          "."
        ]
      }
    ],
    "conflicting_exact_context_collisions": [],
    "neural_instances_created": 0,
    "neural_inference_performed": false,
    "note": "Token-sequence capacity, rawcheckpoint OOV, and exact rendered text are distinct. Spaced decimals can be token-equivalent but fail exact surface output. No codec changed."
  },
  "separate_blockers": {
    "raw_checkpoint_lacks_tokens": true,
    "exact_surface_roundtrip_fails": true,
    "unsupported_capacity": false
  },
  "automatic_learning_allowed": false,
  "main_training": false,
  "pilot_run": false,
  "quality_score_computed": false,
  "decoder_or_encoder_changed": false
}
