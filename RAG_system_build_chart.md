# 🧱 RAGシステム構築チャート（MCPサーバー）

## 1. 環境構築

### 1.1 uvのインストールと仮想環境の作成

```bash
pip install uv
uv venv .venv
uv pip install --upgrade pip
```

### 1.2 必要なパッケージのインストール

```bash
uv pip install langchain chromadb faiss-cpu transformers sentence-transformers pypdf python-docx openpyxl pandas
```

### 1.3 OllamaのセットアップとGemma3の常駐

- Ollamaをインストールし、Gemma3モデルをダウンロードして常駐させます。

---

## 2. ディレクトリ構造の作成

```bash
project_root/
├── data/                  # 元データ（PDF, Word, Excel, CSV, Markdownなど）
├── embeddings/            # ベクトルデータベースの保存先
├── models/                # 埋め込みモデルの保存先
├── scripts/               # 各種スクリプト
│   ├── extract_text.py
│   ├── chunk_text.py
│   ├── embed_chunks.py
│   ├── build_vector_db.py
│   ├── query_vector_db.py
│   └── generate_response.py
└── main.py                # メインスクリプト
```

---

## 3. データ処理フロー

### 3.1 データの読み込みとテキスト抽出
- `extract_text.py`：PDF、Word、Excel、CSV、Markdownファイルからテキストを抽出します。

### 3.2 セマンティックチャンクの生成
- `chunk_text.py`：LangChainの`RecursiveCharacterTextSplitter`を使用して、意味的なまとまりでテキストを分割します。

### 3.3 ベクトル化とデータベースへの格納
- `embed_chunks.py`：intfloat/multilingual-e5-largeモデルを使用してチャンクをベクトル化します。
- `build_vector_db.py`：ベクトル化されたチャンクをFAISSやChromaDBに格納します。

---

## 4. クエリ処理フロー

### 4.1 ユーザープロンプトの処理
- `main.py`：ユーザーからの入力を受け取り、Ollama上のGemma3で解釈と情報の拡張を行います。

### 4.2 ベクトルデータベースからの情報取得
- `query_vector_db.py`：拡張されたプロンプトをベクトル化し、ベクトルデータベースから関連情報を取得します。

### 4.3 応答の生成
- `generate_response.py`：取得した情報をOllama上のGemma3に渡し、最終的な応答を生成します。

---

## 5. 翻訳機能の追加（オプション）
- `translate.py`：生成された応答をユーザーの希望する言語に翻訳します。Google Translate APIやDeepL APIなどを使用できます。

---

## 6. MCPサーバーの構築と限定公開
- MCP（Model Context Protocol）を使用して、構築したRAGシステムを知人限定で公開できます。アクセス制御にはOAuth 2.1やAPIキーを使用します。

---

この構成により、全ての処理がローカルで完結し、プライバシーとセキュリティを確保しながら、高速な情報検索と生成が可能なRAGシステムを構築できます。必要に応じて、各スクリプトの詳細や設定ファイルの構成についてもご案内いたしますので、お気軽にお知らせください。
