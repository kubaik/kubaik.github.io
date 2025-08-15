# 🤖 AI Blog Automation Setup Guide

This guide will help you set up a fully automated AI-powered blog that generates and publishes new content daily using GitHub Actions and GitHub Pages.

## 🚀 Quick Start

### Step 1: Repository Setup

1. **Fork or create this repository**
2. **Enable GitHub Pages**:
   - Go to repository Settings → Pages
   - Set Source to "GitHub Actions"
   - Save the settings

### Step 2: Get OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create a new API key
3. Copy the key (you'll need it in the next step)

### Step 3: Add Repository Secrets

1. Go to your repository Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add these secrets:
   - **Name**: `OPENAI_API_KEY`
   - **Value**: Your OpenAI API key from Step 2

### Step 4: Configure Your Blog

1. **Update config.yaml**:
   ```yaml
   base_url: "https://YOUR-USERNAME.github.io/YOUR-REPO-NAME"
   site_name: "Your Blog Name"
   site_description: "Your blog description"
   ```

2. **Customize topics** (optional):
   - Edit the `content_topics` list in `config.yaml`
   - Add your preferred topics for AI content generation

### Step 5: Trigger First Generation

1. Go to Actions tab in your repository
2. Click on "🤖 AI Blog Automation" workflow
3. Click "Run workflow" → "Run workflow"
4. Wait for the workflow to complete (usually 2-3 minutes)

Your blog will be live at: `https://YOUR-USERNAME.github.io/YOUR-REPO-NAME`

## 📁 Project Structure

```
your-repo/
├── blog_system.py          # Main blog generation system
├── config.yaml            # Blog configuration
├── requirements.txt        # Python dependencies
├── .github/workflows/      # GitHub Actions
│   └── ai-blog.yml
├── docs/                   # Generated static site
│   ├── index.html         # Homepage
│   ├── about/             # About page
│   ├── static/            # CSS and assets
│   ├── [post-slug]/       # Individual blog posts
│   ├── sitemap.xml        # SEO sitemap
│   └── robots.txt         # Search engine instructions
└── .used_topics.json      # Tracks used topics
```

## 🛠 Commands

### Local Development

```bash
# Initialize blog system
python blog_system.py init

# Generate a single post and build site
python blog_system.py auto

# Rebuild site from existing posts
python blog_system.py build
```

### Testing Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key
export OPENAI_API_KEY="your-key-here"

# Generate content
python blog_system.py auto

# Serve locally (optional)
cd docs
python -m http.server 8000
# Visit http://localhost:8000
```

## ⚙️ Configuration Options

### config.yaml

```yaml
site_name: "Your Site Name"
site_description: "Brief description of your blog"
base_url: "https://username.github.io/repo-name"

content_topics:
  - "Your Topic 1"
  - "Your Topic 2"
  - "Add as many as you want"
```

### Environment Variables

- `OPENAI_API_KEY`: Required for content generation
- GitHub Actions automatically provides `GITHUB_TOKEN`

## 🔄 How It Works

1. **Daily Automation**: Runs every day at 9 AM UTC via GitHub Actions
2. **Topic Selection**: Randomly picks an unused topic from your config
3. **AI Generation**: Uses OpenAI's API to generate:
   - SEO-optimized title
   - Comprehensive blog post content (800-1200 words)
   - Meta description
   - Relevant keywords and tags
4. **Static Site Generation**: Converts content to HTML with:
   - Responsive design
   - SEO optimization (sitemap, robots.txt)
   - Clean, modern styling
5. **Deployment**: Automatically deploys to GitHub Pages

## 🎨 Customization

### Styling

Edit the CSS generation in `blog_system.py` (look for `_generate_css` method) or modify the generated `docs/static/style.css` directly.

### Templates

Modify the Jinja2 templates in the `_load_templates` method:
- `base`: Main HTML structure
- `post`: Individual blog post layout
- `index`: Homepage layout
- `about`: About page layout

### Content Generation

Modify the OpenAI prompts in these methods:
- `_generate_title`: Blog post titles
- `_generate_content`: Main blog content
- `_generate_meta_description`: SEO descriptions
- `_generate_keywords`: SEO keywords

## 🐛 Troubleshooting

### Common Issues

1. **"No index.html generated"**:
   - Check if OpenAI API key is valid
   - Ensure config.yaml exists and is properly formatted
   - Check GitHub Actions logs for detailed error messages

2. **"OpenAI API error"**:
   - Verify your API key is correct
   - Check your OpenAI account has available credits
   - API key might need billing setup

3. **GitHub Pages not updating**:
   - Ensure GitHub Pages is set to "GitHub Actions" source
   - Check the Actions tab for deployment status
   - DNS changes can take a few minutes to propagate

4. **Empty or broken site**:
   - Run `python blog_system.py build` locally to test
   - Check that docs/index.html exists after generation
   - Verify config.yaml has correct base_url

### Debug Commands

```bash
# Check generated files
ls -la docs/

# Verify post data
cat docs/your-post-slug/post.json

# Test site generation
python blog_system.py build

# Check workflow logs
# Go to GitHub Actions tab → Click on latest workflow run
```

## 📈 SEO Features

- **Automatic sitemap.xml** generation
- **robots.txt** for search engines
- **Meta descriptions** and keywords
- **Semantic HTML** structure
- **Responsive design** for mobile SEO
- **Fast loading** static site
- **Clean URLs** (e.g., `/post-title/`)

## 🔒 Security

- OpenAI API key stored as GitHub repository secret
- No sensitive data in code repository
- Static site with no server-side vulnerabilities
- GitHub Actions runs in isolated containers

## 💰 Cost Estimation

With GPT-3.5-turbo pricing:
- ~$0.02-0.05 per blog post
- Daily posts = ~$0.60-1.50 per month
- Adjust generation frequency to control costs

## 📞 Support

If you encounter issues:

1. Check the [GitHub Actions logs](../../actions)
2. Verify your configuration against this guide
3. Test locally with the debug commands above
4. Create an issue in this repository with:
   - Error messages from Actions logs
   - Your config.yaml (remove sensitive data)
   - Steps you've already tried

## 🎯 Next Steps

Once your blog is running:

1. **Customize the design** to match your brand
2. **Add analytics** (Google Analytics, etc.)
3. **Set up custom domain** (optional)
4. **Add more content types** (tutorials, reviews, etc.)
5. **Integrate social media** sharing
6. **Add newsletter signup** forms
7. **Optimize for specific niches** in your topics

Happy blogging! 🚀