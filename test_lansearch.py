import requests
import json

url = "https://api.langsearch.com/v1/web-search"

payload = json.dumps({
  "query": "tell me the highlights from Apple's 2024 ESG report",
  "freshness": "noLimit",
  "summary": True,
  "count": 10
})
headers = {
'Authorization': "sk-57a16a19228b4a5897da5e9dd58980ae",
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)

