# Frozen basicdata integrity

Independent source/query/language checks, not neuralquality. No newgeneration or training; retentionfixture not executed.

{
  "run": "runs/20261110T201535Z-basic-knowledge-integrity-v1-ba1133",
  "decision": "FROZEN_POOL_REQUIRES_QUERY_REVIEW",
  "source_data": "data/candidates/20261110T191533Z-basic-knowledge-repair-v1-231806/verified-pool.json",
  "source_sha256": "13bb051467b27caa10bf8d16039d52fa830baf77a5e87bd63e3520d8b96ed062",
  "records": 320,
  "source_facts": 40,
  "issues": [
    {
      "id": "config_adaptation_decay_p04",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "config_adaptation_decay_p06",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "config_adaptation_strength_p01",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "config_adaptation_strength_p02",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "config_adaptation_strength_p03",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "config_anchor_strength_p08",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "config_fan_in_p02",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "config_fan_in_p04",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "config_fan_in_p07",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "config_feature_decay_p08",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "config_hidden_p03",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "config_hidden_p06",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "config_leak_p02",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "config_leak_p05",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "config_leak_p06",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "config_leak_p07",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "config_leak_p08",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "config_outputs_p08",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "config_plastic_rows_p06",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "config_readout_lr_p08",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "config_threshold_p01",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "config_threshold_p04",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "config_threshold_p08",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "config_trace_decay_p08",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "g1_answer_token_cap_p02",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "g1_answer_token_cap_p04",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "g1_answer_token_cap_p08",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "g1_hidden_p06",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "g1_leak_p08",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "g1_max_neurons_p02",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "g1_max_neurons_p04",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "g1_max_neurons_p06",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "g1_plastic_rows_p05",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "g1_vocabulary_cap_p04",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "g1_vocabulary_cap_p06",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    },
    {
      "id": "g1_vocabulary_cap_p08",
      "reason": "SOURCE_VALUE_QUERY_UNVERIFIED"
    }
  ],
  "cross_split_exact_duplicates": [],
  "answer_token_format_variants": {
    "api_chat_route": 2,
    "api_checkpoint_route": 2,
    "api_export_route": 2,
    "api_host": 2,
    "api_model_id": 2,
    "api_models_route": 2,
    "api_port": 2,
    "api_status_route": 2
  },
  "retention_fixture": "runs/20261110T201535Z-basic-knowledge-integrity-v1-ba1133/retention_stream.json",
  "retention_events": 320,
  "retention_stages": 20,
  "retention_executed": false,
  "input_boundary_checked": true,
  "new_teacher_calls": 0,
  "main_training": false,
  "pilot_run": false,
  "final_quality_scores_opened": false,
  "frozen_source_preserved": true
}
