---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:75
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: What are the Manual Services offered at counters of Service Section,
    Examination Branch?|
  sentences:
  - Yes, the campus shuttle operates from 8 am to 8 pm.
  - You should contact JNTUH Exam Branch at Email:support.oss@jntuh.ac.in or at helpline
    no. 9491283135 during 10:30 AM to 5:00 PM on all working days.
  - Issuing of Duplicate Marks Memos, Duplicate CMM, Duplicate Degree Certificate,
    Name & Father Name correction in PC & OD Certificates, Name corrections in marks
    memo & CMM, Issuing of Transcripts, Issuing of PC & CMM with Undertaking and Grace
    Marks.
- source_sentence: How long does it take to get my Certificates after applying in
    online student service?|
  sentences:
  - The Candidates who passed their degree during or after the academic year mentioned
    below can only avail this service. B.Tech?2000 to till date, B.Pharmacy- 2009
    batch to till date, M.Tech ? 2009 batch to till date, M.Pharmacy ?2009 batch to
    till date, MBA ? 2005 batch to till date, MCA - 2005 batch to till date.
  - The printing and dispatching process will be completed within 2 working days after
    the payment is approved, time to time you will receive message / mail once it
    is ready to dispatch from JNTUH
  - PAYMENT APPROVED - After payment approved, PRINTING - Printing of applied certificates
    INPROCESS - checking of applied certificates with address slip DISPATCHEDBYPOST
    ? Dispatched the applied certificates by post DISPATCHEDBYHAND ? Dispatched the
    applied certificates to service counter DELIVERED - the applied certificates received
    by the candidate.
- source_sentence: What are the Online Services offered?|
  sentences:
  - Issuing of Original Degree Certificate, Migration Certificate, Medium of Instruction
    Certificate and issuing Transcripts along with WES Application
  - It is also sent to your registered email id, hence you can login to your mail
    and can take the printout whenever you require. The receipt should be preserved
    for future reference
  - Yes, laptops are allowed in designated areas of the library.
- source_sentence: About JNTUH college?|
  sentences:
  - It is also sent to your registered email id, Hence you can login to your mail
    and can take the printout whenever you require.
  - You should contact JNTUH Exam Branch at Email:support.oss@jntuh.ac.in or at helpline
    no. 9491283135 during 10:30 AM to 5:00 PM on all working days.
  - Jawaharlal Nehru Technological University College of Engineering Hyderabad (Autonomous),
    formerly known as Nagarjuna Sagar Engineering College, was established in 1965
    by the Government of Andhra Pradesh and administrated under the control of the
    Department of Technical Education and affiliated to Osmania University, Hyderabad.
    With the formation of Jawaharlal Nehru Technological University(JNTU) on 2nd October
    1972, the college was made as a constituent college of the University and eventually
    renamed as JNTU College of Engineering, Hyderabad.
- source_sentence: As I am trying to track using the consignment number, it shows
    the consignment number given is not valid.What should I do?|
  sentences:
  - Consignment number received by you is correct. It will be trackable once postal/courier
    service picks and updates their database. You are advised to try again at the
    respective the postal / courier portals on the following day.
  - The late fee is $0.50 per day per book.
  - Fill out the application form at the library reception.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision fa97f6e7cb1a59073dff9e6b13e2715cf7475ac9 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 tokens
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'As I am trying to track using the consignment number, it shows the consignment number given is not valid.What should I do?|',
    'Consignment number received by you is correct. It will be trackable once postal/courier service picks and updates their database. You are advised to try again at the respective the postal / courier portals on the following day.',
    'Fill out the application form at the library reception.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 75 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 75 samples:
  |         | sentence_0                                                                        | sentence_1                                                                         | label                                                         |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                            | string                                                                             | float                                                         |
  | details | <ul><li>min: 7 tokens</li><li>mean: 15.44 tokens</li><li>max: 35 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 34.77 tokens</li><li>max: 246 tokens</li></ul> | <ul><li>min: 1.0</li><li>mean: 1.0</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                     | sentence_1                                                                               | label            |
  |:-------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------|:-----------------|
  | <code> How to apply online transcript for Original Degree certificate?|</code> | <code>At present, there is no transcript service for Original Degree certificate.</code> | <code>1.0</code> |
  | <code>What are the timings of sports complex?|</code>                          | <code>6 am to 8 am and 4:45 pm to 7:00 pm</code>                                         | <code>1.0</code> |
  | <code>Can visitors use the library facilities?|</code>                         | <code>Yes, visitors can use library facilities with a guest pass.</code>                 | <code>1.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Framework Versions
- Python: 3.9.0
- Sentence Transformers: 3.3.1
- Transformers: 4.46.3
- PyTorch: 2.5.0+cpu
- Accelerate: 1.1.1
- Datasets: 3.1.0
- Tokenizers: 0.20.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->