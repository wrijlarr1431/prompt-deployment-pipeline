# Prompt Deployment Pipeline

Automated prompt deployment and content generation pipeline using Amazon Bedrock and S3, orchestrated through GitHub Actions.

## 🎯 Overview

This pipeline enables Pixel Learning Co. to:
- Automate content generation using Amazon Bedrock (Claude 3 Sonnet)
- Manage prompts and templates via version control
- Deploy beta previews on pull requests
- Publish production content on merge to main
- Host generated content on S3 static websites

GitHub PR → Beta Workflow → Bedrock → S3 Beta Bucket
GitHub Merge → Prod Workflow → Bedrock → S3 Prod Bucket


## 📋 Prerequisites

### AWS Resources
1. **S3 Buckets** (2):
   - Beta bucket with static website hosting enabled
   - Prod bucket with static website hosting enabled
   
2. **Amazon Bedrock**:
   - Model access 
   
3. **IAM User**:
   - Permissions for `bedrock:InvokeModel`
   - Permissions for `s3:PutObject` on both buckets

### GitHub
- Repository with Actions enabled
- Configured secrets

## 📝 Usage

### Creating a New Prompt

1. **Create a template** in `prompt_templates/`:
prompt_templates/course_summary.txt

Use `{variable_name}` for placeholders.

2. **Create a config** in `prompts/`:
```json
{
  "template": "course_summary.txt",
  "variables": {
    "course_name": "AI Fundamentals",
    "module_count": "8"
  },
  "model_params": {
    "max_tokens": 1500
  },
  "output_file": "summary_ai_fundamentals.html",
  "instruction": "Format as HTML with proper structure"
}