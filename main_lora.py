from libs import *
from peft import LoraConfig, TaskType
from transformers import BertForSequenceClassification
from peft import get_peft_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/Knowledge_Base/', help='Directory for data dir')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data') #42
    parser.add_argument('--num-classes', type=int, default=4, help='Num of grade')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.00009, help='Learning rate') #0.0001
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU')
    parser.add_argument('--model', type=str, help='Model name or path')
    parser.add_argument('--path', type=str, default= f"/home/leviethai/AI4Sol_Grade/result") #Fix to your path to save model
    parser.add_argument('--gpu', type=int, default=1, help='GPU device')
    parser.add_argument('--eval', type=str, default='test', help='Evaluation on test or valid set')
    
    
    
    return parser.parse_args()



if __name__== "__main__":
    args = parse_args()
    args.best_metric = 0
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}') # Change to your suitable GPU device
        
    # df = pd.read_csv('data/Grade_data.csv')
    # dataset = Dataset.from_pandas(df)
    
    df_train =pd.read_csv('data/Grade_data_train_set.csv')
    df_test =pd.read_csv('data/Grade_data_test_set.csv')
    df_valid =pd.read_csv('data/Grade_data_valid_set.csv')
    
    dataset_train = Dataset.from_pandas(df_train)
    dataset_test = Dataset.from_pandas(df_test)
    dataset_valid = Dataset.from_pandas(df_valid)

    #Login
    if args.model in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Meta-Llama-3-8B-Instruct']:
        from huggingface_hub import login
        login()
    
    # Load the model and tokenizer set up
    model_name=args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    def preprocess_function(examples):
        return tokenizer(examples["Question"], truncation=True)
    
    
    tokenized_dataset_train = dataset_train.map(preprocess_function, batched=True)
    tokenized_dataset_test = dataset_test.map(preprocess_function, batched=True)
    tokenized_dataset_valid = dataset_valid.map(preprocess_function, batched=True)
    
    
    
    # Lora Setup
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4, id2label=id2label, label2id=label2id)
    
    print('The original model: ')
    print(print_number_of_trainable_model_parameters(model))
    
    lora_config = LoraConfig(
        r=32, # Rank
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["o_proj", "qkv_proj"],
        task_type=TaskType.SEQ_CLS
    )
    model = get_peft_model(model, lora_config)

    print('The model after adding Lora: ')
    print(print_number_of_trainable_model_parameters(model))
    
    
    
    # Training setup
    training_args = TrainingArguments(
    output_dir = args.path,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_valid,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    if args.phase == 'train':
        trainer.train()
        
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Save the trained model with timestamp prefix
        model_output_dir = os.path.join(args.path, args.model, current_time)
        
        # model.save_pretrained(model_output_dir)
        # tokenizer.save_pretrained(model_output_dir)
        
        trainer.save_model(model_output_dir)
        
        print(f"Model saved to {model_output_dir}")
        
        print("Evaluation on test set...")
        eval_results = trainer.evaluate(eval_dataset=tokenized_dataset_test)
        print(eval_results)
    
    elif args.phase == 'test':
        if args.eval == 'valid':
            print("Evaluation on valid set...")
            eval_results = trainer.evaluate(eval_dataset=tokenized_dataset_valid)
            print(eval_results)
            
        elif args.eval == 'test':
            print("Evaluation on test set...")
            eval_results = trainer.evaluate(eval_dataset=tokenized_dataset_test)
            print(eval_results)
            
        elif args.eval == 'train':
            print("Evaluation on train set...")
            eval_results = trainer.evaluate(eval_dataset=tokenized_dataset_train)
            print(eval_results)

    