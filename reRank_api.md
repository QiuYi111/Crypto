---
description: Improve the accuracy of your search results with semantic reranking.
---

# Semantic Rerank API

## **1. API Overview**

The **Semantic Rerank API** is designed to improve the accuracy of your search results by leveraging deep semantic understanding to reorder results based on the relevance between the search query and the document content. This API performs a second-layer ranking on the initial search results (e.g., from keyword or vector search) to ensure that the most semantically relevant documents are presented to the user.

## **2. Rank Results**

The Semantic Rerank API evaluates the semantic match between the search query and a set of candidate documents. It then reorders the results, assigning each document a semantic relevance score. This process is designed to enhance traditional ranking mechanisms like BM25 or RRF by considering the deeper semantic meaning behind both the query and document content.

* **index**: The position of the document in the document list.
* **document**: The document details, including:
  * **text**: The document content.
* **relevance\_score**: The semantic relevance score, ranging from 0 to 1. A higher score indicates stronger semantic alignment with the query.

## **3. API Endpoint**

* **API Domain**: `https://api.langsearch.com`
* **Endpoint**: `https://api.langsearch.com/v1/rerank`

## **4. Request Method**

* **Method**: `POST`

## **5. Request Parameters**

**Request Headers**

<table><thead><tr><th width="191">Parameter</th><th width="196">Value</th><th>Description</th></tr></thead><tbody><tr><td><code>Authorization</code></td><td><code>Bearer {API KEY}</code></td><td>Authentication parameter. Example: <code>Bearer xxxxxx</code>. API KEY can be obtained from the LangSearch API Dashboard (<a href="https://langsearch.com/dashboard">https://langsearch.com/dashboard</a>) > API Key Management.</td></tr><tr><td><code>Content-Type</code></td><td><code>application/json</code></td><td>Specifies the format of the request body.</td></tr></tbody></table>

**Request Body:**

<table><thead><tr><th width="192">Parameter</th><th width="143">Type</th><th width="110">Required</th><th>Description</th></tr></thead><tbody><tr><td><strong>model</strong></td><td>String</td><td>Yes</td><td>The model version to use for reranking. Available models:<br>- langsearch-reranker-v1 </td></tr><tr><td><strong>query</strong></td><td>String</td><td>Yes</td><td>The search query (can be in natural language, e.g., "Tell me the key points of Alibaba's 2024 ESG report").</td></tr><tr><td><strong>documents</strong></td><td>Array&#x3C;String></td><td>Yes</td><td>Array of documents to be reranked (max 50 documents).</td></tr><tr><td><strong>top_n</strong></td><td>Integer</td><td>No</td><td>The number of top-ranked documents to return. Default is the number of documents provided.</td></tr><tr><td><strong>return_documents</strong></td><td>Boolean</td><td>No</td><td>Whether to return the original text of each document in the result. Default is <code>false</code>.</td></tr></tbody></table>

**Response:**

<table><thead><tr><th width="157">Parameter</th><th width="106">Type</th><th>Description</th></tr></thead><tbody><tr><td><strong>code</strong></td><td>Integer</td><td>Status code. 200 means success.</td></tr><tr><td><strong>log_id</strong></td><td>String</td><td>Request ID for tracking.</td></tr><tr><td><strong>msg</strong></td><td>String</td><td>Status message.</td></tr><tr><td><strong>model</strong></td><td>String</td><td>The model used for reranking.</td></tr><tr><td><strong>results</strong></td><td>Array</td><td>The reranked results. Each result includes the document index and its relevance score. </td></tr></tbody></table>

## **6. Example Request**

**Request Body**:

```json
{
    "model": "langsearch-reranker-v1",
    "query": "Tell me the key points of Alibaba's 2024 ESG report",
    "top_n": 2,
    "return_documents": true,
    "documents": [
        "Alibaba Group released the 2024 Environmental, Social, and Governance (ESG) report, detailing the progress made in various ESG areas over the past year. The report shows that Alibaba has steadily advanced its carbon reduction efforts, with the group's net carbon emissions and carbon intensity of the value chain continuing to decrease. The group also continues to leverage digital technologies and platform capabilities to support accessible development, healthcare, aging-friendly services, and small and micro enterprises. Alibaba Group's CEO, Wu Yongming, stated in the report: 'The core of ESG is about becoming a better company. Over the past 25 years, the actions related to ESG have formed the foundation of Alibaba, which is just as important as the commercial value we create. While the group is focused on the dual business strategies of 'user-first' and 'AI-driven,' we also remain committed to ESG as one of Alibaba's cornerstone strategies. Alibaba has made solid progress in reducing carbon emissions.'",
        "The core of ESG is about becoming a better company. This year marks the 25th anniversary of Alibaba. Over the past 25 years, Alibaba has adhered to its mission of 'making it easy to do business everywhere,' supporting the prosperity of domestic e-commerce; maintaining an open ecosystem, with the Magic搭 community opening over 3,800 open-source models; assisting in rural revitalization, having sent 29 rural special envoys to 27 counties; promoting platform carbon reduction, pioneering a Scope 3+ carbon reduction plan; and engaging in employee welfare, with the 'Everyone 3 Hours' initiative making small but meaningful changes... These actions form the foundation of Alibaba, which is just as important as creating commercial value. I hope that every Alibaba employee will learn to make difficult but correct choices, maintaining foresight, goodwill, and pragmatism. A better Alibaba is worth our collective efforts. Alibaba's mission, unchanged for over 20 years, is to make it easy to do business in the world. Today, this mission takes on new significance in this era."
    ]
}
```

**Response Body**:

```json
{
    "code": 200,
    "log_id": "56a3067f9b92dfd0",
    "msg": null,
    "model": "langsearch-reranker-v1",
    "results": [
        {
            "index": 0,
            "document": {
                "text": "Alibaba Group released the 2024 Environmental, Social, and Governance (ESG) report, detailing the progress made in various ESG areas over the past year. The report shows that Alibaba has steadily advanced its carbon reduction efforts, with the group's net carbon emissions and carbon intensity of the value chain continuing to decrease. The group also continues to leverage digital technologies and platform capabilities to support accessible development, healthcare, aging-friendly services, and small and micro enterprises. Alibaba Group's CEO, Wu Yongming, stated in the report: 'The core of ESG is about becoming a better company. Over the past 25 years, the actions related to ESG have formed the foundation of Alibaba, which is just as important as the commercial value we create. While the group is focused on the dual business strategies of 'user-first' and 'AI-driven,' we also remain committed to ESG as one of Alibaba's cornerstone strategies. Alibaba has made solid progress in reducing carbon emissions.'"
            },
            "relevance_score": 0.7166407801262326
        },
        {
            "index": 1,
            "document": {
                "text": "The core of ESG is about becoming a better company. This year marks the 25th anniversary of Alibaba. Over the past 25 years, Alibaba has adhered to its mission of 'making it easy to do business everywhere,' supporting the prosperity of domestic e-commerce; maintaining an open ecosystem, with the Magic搭 community opening over 3,800 open-source models; assisting in rural revitalization, having sent 29 rural special envoys to 27 counties; promoting platform carbon reduction, pioneering a Scope 3+ carbon reduction plan; and engaging in employee welfare, with the 'Everyone 3 Hours' initiative making small but meaningful changes... These actions form the foundation of Alibaba, which is just as important as creating commercial value. I hope that every Alibaba employee will learn to make difficult but correct choices, maintaining foresight, goodwill, and pragmatism. A better Alibaba is worth our collective efforts. Alibaba's mission, unchanged for over 20 years, is to make it easy to do business in the world. Today, this mission takes on new significance in this era."
            },
            "relevance_score": 0.5658672473649548
        }
    ]
}
```

## 7. SDKs

### cURL

```bash
curl --location 'https://api.langsearch.com/v1/rerank' \
--header 'Authorization: Bearer YOUR-API-KEY' \
--header 'Content-Type: application/json' \
--data '{
    "model": "langsearch-reranker-v1",
    "query": "Tell me the key points of Alibaba 2024 ESG report",
    "top_n": 2,
    "return_documents": true,
    "documents": [
        "Alibaba Group released the 2024 Environmental, Social, and Governance (ESG) report, detailing the progress made in various ESG areas over the past year. The report shows that Alibaba has steadily advanced its carbon reduction efforts, with the group'\''s net carbon emissions and carbon intensity of the value chain continuing to decrease. The group also continues to leverage digital technologies and platform capabilities to support accessible development, healthcare, aging-friendly services, and small and micro enterprises. Alibaba Group'\''s CEO, Wu Yongming, stated in the report: '\''The core of ESG is about becoming a better company. Over the past 25 years, the actions related to ESG have formed the foundation of Alibaba, which is just as important as the commercial value we create. While the group is focused on the dual business strategies of '\''user-first'\'' and '\''AI-driven,'\'' we also remain committed to ESG as one of Alibaba'\''s cornerstone strategies. Alibaba has made solid progress in reducing carbon emissions.'\''",
        "The core of ESG is about becoming a better company. This year marks the 25th anniversary of Alibaba. Over the past 25 years, Alibaba has adhered to its mission of '\''making it easy to do business everywhere,'\'' supporting the prosperity of domestic e-commerce; maintaining an open ecosystem, with the Magic搭 community opening over 3,800 open-source models; assisting in rural revitalization, having sent 29 rural special envoys to 27 counties; promoting platform carbon reduction, pioneering a Scope 3+ carbon reduction plan; and engaging in employee welfare, with the '\''Everyone 3 Hours'\'' initiative making small but meaningful changes... These actions form the foundation of Alibaba, which is just as important as creating commercial value. I hope that every Alibaba employee will learn to make difficult but correct choices, maintaining foresight, goodwill, and pragmatism. A better Alibaba is worth our collective efforts. Alibaba'\''s mission, unchanged for over 20 years, is to make it easy to do business in the world. Today, this mission takes on new significance in this era."
    ]
}'
```

### Python

```python
import requests
import json

url = "https://api.langsearch.com/v1/rerank"

payload = json.dumps({
  "model": "langsearch-reranker-v1",
  "query": "Tell me the key points of Alibaba 2024 ESG report",
  "top_n": 2,
  "return_documents": True,
  "documents": [
    "Alibaba Group released the 2024 Environmental, Social, and Governance (ESG) report, detailing the progress made in various ESG areas over the past year. The report shows that Alibaba has steadily advanced its carbon reduction efforts, with the group's net carbon emissions and carbon intensity of the value chain continuing to decrease. The group also continues to leverage digital technologies and platform capabilities to support accessible development, healthcare, aging-friendly services, and small and micro enterprises. Alibaba Group's CEO, Wu Yongming, stated in the report: 'The core of ESG is about becoming a better company. Over the past 25 years, the actions related to ESG have formed the foundation of Alibaba, which is just as important as the commercial value we create. While the group is focused on the dual business strategies of 'user-first' and 'AI-driven,' we also remain committed to ESG as one of Alibaba's cornerstone strategies. Alibaba has made solid progress in reducing carbon emissions.'",
    "The core of ESG is about becoming a better company. This year marks the 25th anniversary of Alibaba. Over the past 25 years, Alibaba has adhered to its mission of 'making it easy to do business everywhere,' supporting the prosperity of domestic e-commerce; maintaining an open ecosystem, with the Magic搭 community opening over 3,800 open-source models; assisting in rural revitalization, having sent 29 rural special envoys to 27 counties; promoting platform carbon reduction, pioneering a Scope 3+ carbon reduction plan; and engaging in employee welfare, with the 'Everyone 3 Hours' initiative making small but meaningful changes... These actions form the foundation of Alibaba, which is just as important as creating commercial value. I hope that every Alibaba employee will learn to make difficult but correct choices, maintaining foresight, goodwill, and pragmatism. A better Alibaba is worth our collective efforts. Alibaba's mission, unchanged for over 20 years, is to make it easy to do business in the world. Today, this mission takes on new significance in this era."
  ]
})
headers = {
  'Authorization': 'Bearer YOUR-API-KEY',
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
```

