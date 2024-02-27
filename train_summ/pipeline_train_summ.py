from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer


if __name__ == "__main__":

    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    training_args = Seq2SeqTrainingArguments(
        output_dir="my_fine_tuned_t5_small_model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=4,
        predict_with_generate=True,
        fp16=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_billsum["train"],
        eval_dataset=tokenized_billsum["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )