# Publish HookRec to GitHub

The project is ready to initialize locally. If GitHub CLI (`gh`) is installed and authenticated, run these commands from the extracted `hookrec` directory:

```bash
git init
git add .
git commit -m "Build HookRec recommendation MVP"
gh repo create hookrec --public --source=. --remote=origin --push
```

If you prefer the GitHub website, create an empty public repository named `hookrec` (do not initialize it with a README or license), then run:

```bash
git init
git add .
git commit -m "Build HookRec recommendation MVP"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/hookrec.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. Do not commit the downloaded dataset or generated evaluation report; both are excluded by `.gitignore`.
