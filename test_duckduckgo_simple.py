def test_duckduckgo(query):
    from ddgs import DDGS

    ddgs = DDGS()
    try:
        results = ddgs.text(query)
        print(results)  # Inspect the results
    except Exception as e:
        print(f"An error occurred: {e}")

    for result in results:
        print(result['title'], result['href'])

if __name__ == "__main__":
    test_duckduckgo("BTC")

