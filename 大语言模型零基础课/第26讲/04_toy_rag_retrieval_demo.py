# 第26讲实验：Toy RAG Retrieval
# 纯 Python，无第三方 NLP 库。
# 演示：Chunk -> TF-IDF -> Cosine -> Top-k -> Recall@k
#
# 注意：真实 RAG 常使用专门的 embedding model + vector index。
# 这里用 TF-IDF 是为了把检索过程看清楚。

import math
import re
from collections import Counter, defaultdict

documents = {
    "doc_api": (
        "When an API returns HTTP 429, the client has hit a rate limit. "
        "Use exponential backoff before retrying. "
        "Respect the Retry-After header when it is present. "
        "Do not retry immediately in a tight loop."
    ),
    "doc_timeout": (
        "A timeout means the request took too long. "
        "Set a reasonable timeout and retry only when the operation is safe. "
        "A timeout is different from an HTTP 429 rate limit."
    ),
    "doc_memory": (
        "Agent memory stores useful past experience outside model parameters. "
        "A retrieval policy selects which memories enter the current context. "
        "Old or unreliable memories should be down-weighted."
    ),
    "doc_kv": (
        "KV cache stores key and value tensors computed for previous tokens. "
        "It reduces repeated computation during autoregressive decoding. "
        "KV cache is not long-term agent memory."
    ),
    "doc_rag": (
        "Retrieval-augmented generation first retrieves relevant evidence. "
        "The evidence is then placed into the model context. "
        "Answer faithfulness measures whether the answer is supported by evidence."
    ),
}

def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def make_chunks(documents, sentences_per_chunk=2, overlap=1):
    chunks=[]
    for doc_id,text in documents.items():
        sents=split_sentences(text)
        step=max(1,sentences_per_chunk-overlap)
        for start in range(0,len(sents),step):
            part=sents[start:start+sentences_per_chunk]
            if not part:
                continue
            chunk_id=f"{doc_id}_c{start}"
            chunks.append((chunk_id,doc_id," ".join(part)))
            if start+sentences_per_chunk>=len(sents):
                break
    return chunks

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def build_tfidf(chunks):
    docs_tokens=[tokenize(text) for _,_,text in chunks]
    N=len(docs_tokens)
    df=Counter()
    for toks in docs_tokens:
        for term in set(toks):
            df[term]+=1
    idf={term: math.log((N+1)/(freq+1))+1.0 for term,freq in df.items()}

    vectors=[]
    for toks in docs_tokens:
        tf=Counter(toks)
        total=max(1,len(toks))
        vec={term:(cnt/total)*idf.get(term,1.0) for term,cnt in tf.items()}
        vectors.append(vec)
    return idf,vectors

def query_vector(query,idf):
    toks=tokenize(query)
    tf=Counter(toks)
    total=max(1,len(toks))
    return {term:(cnt/total)*idf.get(term, math.log(2)+1) for term,cnt in tf.items()}

def cosine(a,b):
    if not a or not b:
        return 0.0
    dot=sum(v*b.get(k,0.0) for k,v in a.items())
    na=math.sqrt(sum(v*v for v in a.values()))
    nb=math.sqrt(sum(v*v for v in b.values()))
    return dot/(na*nb) if na and nb else 0.0

chunks=make_chunks(documents,sentences_per_chunk=2,overlap=1)
idf,vectors=build_tfidf(chunks)

query="HTTP 429 rate limit exponential backoff Retry-After"
qv=query_vector(query,idf)

ranked=[]
for chunk,vec in zip(chunks,vectors):
    cid,doc_id,text=chunk
    ranked.append((cosine(qv,vec),cid,doc_id,text))
ranked.sort(reverse=True,key=lambda x:x[0])

# 人工标注：前两条 api chunks 都算 relevant
relevant={cid for _,cid,doc_id,_ in ranked if doc_id=="doc_api"}

print("Query:",query)
print("="*88)
for i,(score,cid,doc_id,text) in enumerate(ranked[:6],1):
    mark="RELEVANT" if cid in relevant else ""
    print(f"{i}. score={score:.3f}  {cid:18s} {mark}")
    print("   ",text)

print()
for k in [1,2,3,5]:
    top={cid for _,cid,_,_ in ranked[:k]}
    recall=len(top & relevant)/len(relevant)
    precision=len(top & relevant)/k
    print(f"k={k}: Recall@k={recall:.3f}  Precision@k={precision:.3f}")

print()
print("Lesson:")
print("- Larger k can improve recall.")
print("- But larger k can also introduce more irrelevant context.")
print("- A reranker can keep a larger candidate pool while shrinking final context.")
