# Email Classifier

[![GitHub](https://img.shields.io/github/license/cpknight/email-classifier)](https://github.com/cpknight/email-classifier/blob/main/LICENSE)

A machine learning tool that automatically classifies emails into categories such as "notifications" and "correspondence" based on their content and metadata.

## Features

- **Training**: Train the classifier using your own categorized emails
- **Evaluation**: Evaluate the classifier's performance with test data
- **Classification**: Classify new emails automatically
- **Model Backups**: Automatically creates backups of previous models when training new ones

## Installation

### Prerequisites

- Python 3.6 or higher
- pip (Python package installer)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/cpknight/email-classifier.git
   cd email-classifier
   ```

2. Install the required dependencies:
   ```bash
   pip install pandas scikit-learn nltk
   ```

## Usage

The email classifier script supports three main commands: `train`, `evaluate`, and `classify`.

### Training the classifier

```bash
./email_classifier.py train --dir <training_directory> --output <model_file>
```

**Example:**
```bash
./email_classifier.py train --dir ./emails --output model.pkl
```

By default, the script expects your training data to be organized in two subdirectories:
- `notifications/`: Contains notification-type emails
- `correspondence/`: Contains correspondence-type emails

When training a new model, if a model already exists at the specified output path, a backup will be created with the original creation timestamp appended to the filename.

### Evaluating the classifier

```bash
./email_classifier.py evaluate --model <model_file> --dir <test_directory>
```

**Example:**
```bash
./email_classifier.py evaluate --model model.pkl --dir ./test_emails
```

### Classifying emails

```bash
./email_classifier.py classify --model <model_file> --email <email_file>
```

**Example:**
```bash
./email_classifier.py classify --model model.pkl --email ./new_email.eml
```

## Data Organization

For training and evaluation, the script expects emails to be organized in directories by category:

```
training_data/
├── correspondence/
│   ├── email1.eml
│   ├── email2.eml
│   └── ...
└── notifications/
    ├── email1.eml
    ├── email2.eml
    └── ...
```

Each `.eml` file should be a standard email file with headers and content.

## Model Backup Feature

When training a new model, if a model already exists at the specified location, the script automatically creates a backup of the existing model before overwriting it. The backup filename includes the original creation timestamp in the format:

```
model.pkl.YYYYMMDD_HHMMSS.bak
```

This feature ensures you never lose previous model versions and can track model evolution over time.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Created by ~[cpknight](https://github.com/cpknight)~ ... I can't take credit. Claude `3.7 sonnet` by way of Warp wrote it - this is another AI-generated project!

