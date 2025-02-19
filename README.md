# Multi-Agent AI system using CrewAI
In this repository, I've implemented Multi-Agent AI system using CrewAI.
Following are the steps to implement this Multi-Agent AI system.

### 1. Clone Repo
```
git clone https://github.com/MahrukhAliKhan/Multi-Agent-AI-System.git
```

### 2. Create Virtual Environment
```
sudo apt install python3.10-venv
python3.10 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

### 4. Set API-Keys in code
- Generate your own Server Dev API-Key from <a href="https://serper.dev/api-key">here</a>
- Generate your own Hugging-Face API-Key from <a href="https://huggingface.co/settings/tokens">here</a>
Replace your own API-keys in code line # 8,9
```
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<Your HUGGING_FACE API KEY HERE>"
os.environ["SERPER_API_KEY"] = "<YOUR SERPER API KEY HERE>"
```

### 5. Run Multi-Agent Code
```
python multi-agent.py
```


