import json
import os
import sys
import boto3
from pathlib import Path

def load_template(template_name):
    """Load a prompt template from the prompt_templates directory."""
    template_path = Path('prompt_templates') / template_name
    with open(template_path, 'r') as f:
        return f.read()

def render_template(template, variables):
    """Replace variables in template with actual values."""
    rendered = template
    for key, value in variables.items():
        placeholder = f"{{{key}}}"
        rendered = rendered.replace(placeholder, str(value))
    return rendered

def invoke_bedrock(prompt, instruction, model_params):
    """Invoke Amazon Bedrock with the given prompt."""
    bedrock = boto3.client('bedrock-runtime', region_name=os.environ.get('AWS_REGION', 'us-east-1'))
    
    # Construct the full prompt with instruction
    full_prompt = f"{instruction}\n\nContent to transform:\n{prompt}"
    
    # Prepare the request body for Claude 3
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": model_params.get('max_tokens', 2000),
        "temperature": model_params.get('temperature', 0.7),
        "messages": [
            {
                "role": "user",
                "content": full_prompt
            }
        ]
    }
    
    # Invoke the model
    response = bedrock.invoke_model(
        modelId='anthropic.claude-3-sonnet-20240229-v1:0',
        body=json.dumps(body)
    )
    
    # Parse response
    response_body = json.loads(response['body'].read())
    return response_body['content'][0]['text']

def save_output(content, filename, local_dir='outputs'):
    """Save generated content locally."""
    Path(local_dir).mkdir(exist_ok=True)
    output_path = Path(local_dir) / filename
    with open(output_path, 'w') as f:
        f.write(content)
    print(f"Saved locally: {output_path}")
    return output_path

def upload_to_s3(file_path, bucket_name, s3_key):
    """Upload file to S3 bucket."""
    s3 = boto3.client('s3', region_name=os.environ.get('AWS_REGION', 'us-east-1'))
    
    try:
        s3.upload_file(
            str(file_path),
            bucket_name,
            s3_key,
            ExtraArgs={'ContentType': 'text/html'}
        )
        print(f"Uploaded to S3: s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"S3 upload failed: {e}")
        raise

def process_prompt_file(prompt_file, environment):
    """Process a single prompt configuration file."""
    print(f"\n{'='*60}")
    print(f"Processing: {prompt_file}")
    print(f"Environment: {environment.upper()}")
    print(f"{'='*60}\n")
    
    # Load prompt configuration
    with open(prompt_file, 'r') as f:
        config = json.load(f)
    
    # Load and render template
    template = load_template(config['template'])
    rendered_prompt = render_template(template, config['variables'])
    
    print(f"📝 Template: {config['template']}")
    print(f"🔧 Variables: {list(config['variables'].keys())}")
    
    # Invoke Bedrock
    print(f"🤖 Invoking Bedrock...")
    generated_content = invoke_bedrock(
        rendered_prompt,
        config['instruction'],
        config['model_params']
    )
    
    # Save locally
    output_file = save_output(generated_content, config['output_file'])
    
    # Upload to S3
    if environment == 'beta':
        bucket = os.environ.get('S3_BUCKET_BETA')
        s3_key = f"beta/outputs/{config['output_file']}"
    else:  # prod
        bucket = os.environ.get('S3_BUCKET_PROD')
        s3_key = f"prod/outputs/{config['output_file']}"
    
    if bucket:
        upload_to_s3(output_file, bucket, s3_key)
    else:
        print(f"⚠️  No S3 bucket configured for {environment}")
    
    print(f"\n✅ Successfully processed {prompt_file}\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_prompt.py <beta|prod>")
        sys.exit(1)
    
    environment = sys.argv[1].lower()
    if environment not in ['beta', 'prod']:
        print("Environment must be 'beta' or 'prod'")
        sys.exit(1)
    
    # Find all prompt files
    prompts_dir = Path('prompts')
    prompt_files = list(prompts_dir.glob('*.json'))
    
    if not prompt_files:
        print("No prompt files found in prompts/ directory")
        sys.exit(1)
    
    print(f"\n🚀 Starting prompt processing for {environment.upper()} environment")
    print(f"Found {len(prompt_files)} prompt file(s)\n")
    
    # Process each prompt file
    for prompt_file in prompt_files:
        try:
            process_prompt_file(prompt_file, environment)
        except Exception as e:
            print(f"Error processing {prompt_file}: {e}")
            sys.exit(1)
    
    print(f"\n🎉 All prompts processed successfully for {environment.upper()}!\n")

if __name__ == "__main__":
    main()