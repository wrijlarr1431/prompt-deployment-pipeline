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


3. Create a pull request:

git checkout -b add-course-summary
git add prompts/ prompt_templates/
git commit -m "Add course summary prompt"
git push origin add-course-summary

4. Review beta output from the PR comment link

5. Merge to deploy to production

Workflow Triggers
Event	Workflow	Environment	S3 Prefix
Pull Request	on_pull_request.yml	Beta	beta/
Merge to main	on_merge.yml	Production	prod/

🔍 Viewing Generated Content
Beta
http://[S3_BUCKET_BETA].s3-website-[AWS_REGION].amazonaws.com/beta/outputs/
Production
http://[S3_BUCKET_PROD].s3-website-[AWS_REGION].amazonaws.com/prod/outputs/

📁 Project Structure
.
├── .github/
│   └── workflows/
│       ├── on_pull_request.yml    # Beta deployment
│       └── on_merge.yml            # Prod deployment
├── prompts/                        # Prompt configurations
│   └── welcome_prompt.json
├── prompt_templates/               # Reusable templates
│   └── welcome_email.txt
├── outputs/                        # Generated content (local)
├── scripts/
│   └── process_prompt.py          # Main processing script
├── requirements.txt
└── README.md

🔒 Security Notes
Never commit AWS credentials
Use GitHub secrets for all sensitive values
IAM user should have minimal required permissions
S3 buckets use public read for static hosting only

🐛 Troubleshooting
Bedrock Access Denied
Verify model access is enabled in Bedrock console
Check IAM permissions include bedrock:InvokeModel
S3 Upload Failed
Verify bucket names in GitHub secrets
Check IAM permissions include s3:PutObject
Workflow Not Triggering
Ensure changes are in prompts/, prompt_templates/, or scripts/
Check workflow file syntax
