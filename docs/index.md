# StockChat: A Stock Market Assistant

### At a minimum you will need a free-tier api key for

- Google Gemini which you can generate in AI Studio
- Finnhub.io
- Polygon.io

# Laying the foundation with Gemini 2.0

A programming instructor once suggested the idea of a Stock Market application for final project topics. They did this knowing good investing app UX is challenging. The idea has stuck with me since because it's true. In the past I've worked with some REST api's building toys. None of them could ever reach my expectations because of API limits. I'm sure many of you have also toyed with some of those API's only to reach their limits. I always knew the secret to great finance UX is a great AI to help out. When posed with so many topics for 2025's 5-Day GenAI Course, I first tinkered with many of the other capabilities of Gemini until I posed Gemini the question:

```python
# This is an accurate retelling of events. 
config_with_search = types.GenerateContentConfig(
    tools=[types.Tool(google_search=types.GoogleSearch())],
    temperature=0.0
)

chat = api.client.chats.create(
    model=api(Gemini.Model.GEN), 
    config=config_with_search, 
    history=[]) # Ignoring the part about dark elves, and tengwar.

response = chat.send_message('Do you know anything about the stock market?')
Markdown(response.text)
```

>Yes, I do. Here's some information about the stock market:
>
>**What it is:**
>
>*   The stock market is a network of marketplaces (exchanges) where stocks and other investments are bought and sold. It's not a single physical location.
>*   It connects investors who want to buy stocks with those who want to sell.
>*   Stocks, also known as equities, represent partial ownership of a publicly traded company. When you buy a company's stock, you're buying a small piece of that company.
>
>**How it works:**
>
>*   **Initial Public Offering (IPO):** Private companies can list shares of their stock on an exchange through an IPO to raise capital to grow their business.
>*   **Buying and Selling:** Once a company is public, investors can buy and sell shares of that company's stock on an exchange.
>*   **Supply and Demand:** Stock prices are determined by supply and demand. If demand from buyers is greater than the supply from sellers, the price goes up. If there are more sellers than buyers, the price goes down.
>*   **Exchanges:** Stock exchanges are where stocks and other investments are traded. Examples include the New York Stock Exchange (NYSE) and the Nasdaq. These exchanges track the supply and demand of stocks listed there.
>*   **Market Indexes:** Market activity is tracked through indexes like the S&P 500 and the Dow Jones Industrial Average (DJIA).
>
>**Purpose of the stock market:**
>
>*   **Raising Capital:** It helps companies raise money (capital) by selling shares, which they can use to fund and expand their businesses.
>*   **Investment Opportunities:** It provides investors with the opportunity to own a share of a company's profits and potentially grow their wealth.
>*   **Liquidity:** The stock market allows investors to quickly and easily sell their securities.
>
>**Investing in the stock market:**
>
>*   **Potential for Returns:** Investments can increase in value over time as companies grow. Stocks have the potential for higher returns compared to bonds.
>*   **Dividends:** Some stocks pay dividends (a regular payment per share) to shareholders.
>*   **Risk:** There's also the potential for investments to decrease in value, depending on the economy and the specific investment.
>*   **Ways to Invest:** You can buy and sell shares through investment firms or online brokers. You can also invest in funds like mutual funds or Exchange-Traded Funds (ETFs), which hold a variety of stocks or other investments.
>
>**Important considerations:**
>
>*   **Research:** Before investing in a stock, it's a good idea to research the company and its performance history.
>*   **Diversification:** Diversifying your investments (investing in a variety of stocks or funds) can help reduce risk.
>*   **Regulation:** The stock market is regulated by bodies like the U.S. Securities and Exchange Commission (SEC) to protect investors and ensure fair practices.


# How much Gemini 2.0 knows

I thought to myself: Could grounding really make it that easy? Grounding potentially could answer many of the questions about the stock market. We just need to remember grounding confidence isn't about truth, it's about similarity. I decided to limit myself to free tier in finding out.

```python
# And so I asked a more challenging questions.
response = chat.send_message('I have an interest in AMZN stock')
Markdown(response.text)
```

>Okay, here's some information regarding AMZN (Amazon) stock as of July 7, 2025:
>
>**Current Analyst Ratings and Price Targets:**
>
>*   **Consensus:** The consensus rating from analysts is generally "Moderate Buy" or "Strong Buy".
>*   **Average Price Target:** The consensus average price target varies among analysts, but it generally ranges from $243 to $246. This suggests a potential upside of around 9% to 10% from the current price. Some analysts have higher price targets, >reaching up to $305.
>*   **Analyst Recommendations:** A large majority of analysts (over 90%) have a "Buy" or "Strong Buy" rating on the stock.
>
>**Stock Price Forecasts:**
>
>*   **Year 2025:** Forecasts for the remainder of 2025 vary. Some predict a trading channel between $198 and $231, with an average price around $213. Others are more bullish, with some forecasts reaching $260+ by the end of the year.
>*   **Long-Term (2030):** Longer-term forecasts estimate a price of around $430 by the end of 2030.
>
>**Factors to Consider:**
>
>*   **Analyst Updates:** Keep in mind that ratings and forecasts are frequently updated based on market conditions, earnings reports, and industry trends.
>*   **E-commerce:** While e-commerce is still growing, increased competition may make it harder to achieve the same growth rates as in the past.
>*   **Amazon Web Services (AWS):** AWS is a major profit driver for Amazon, and its continued growth in cloud and AI infrastructure is important.
>*   **AI Initiatives:** Amazon's investment in AI, including its own AI model, is a key area to watch.
>*   **Prime Day:** Amazon's Prime Day sales event is expected to generate significant revenue. The 2025 Prime Day is expected to generate $23.8 billion in U.S. online sales.
>*   **Rural Expansion:** Amazon is investing heavily in expanding delivery services to rural areas.
>*   **Potential Risks:** Regulatory concerns and increasing competition are potential risks for Amazon.
>
>**Recent News:**
>
>*   **Investment in AI Data Centers:** Amazon is investing billions in AI data centers in Pennsylvania and North Carolina.
>*   **Prime Delivery Expansion:** Amazon is investing billions to expand Prime delivery services to rural America.
>*   **Jeff Bezos Stock Sales:** Jeff Bezos has a plan to sell a significant number of Amazon shares.
>
>**In summary:** Analysts generally have a positive outlook on Amazon (AMZN) stock, with "Buy" ratings and price targets suggesting potential upside. However, forecasts vary, and it's important to consider factors like competition, AWS performance, >and Amazon's investments in AI.


Impressed, I was reminded of the dreaded REST api's (some official) that I've worked in the past. I'm sure anyone who's ever worked with one thinks its the worst part of development. So I next asked Gemini to distill it's vast news knowledge.

```python
response = chat.send_message(
    '''Tell me about AMZN current share price, short-term trends, and bullish versus bearish predictions''')
Markdown(response.text)
```

>Okay, here's a summary of the AMZN (Amazon) stock situation as of July 7, 2025, based on the information I found:
>
>**Current Share Price:**
>
>*   The price is fluctuating around **$223 - $223.55**.
>*   For example, one source indicates a price of $223.01, reflecting a +1.41% move since the market opened. Another source shows $223.11, up 1.59% in the past 24 hours. Yet another shows $223.46.
>*   Recent trading has seen the stock in a range between approximately $220 and $224.
>
>**Short-Term Trends:**
>
>*   **Generally Bullish:** Many sources indicate a bullish short-term trend.
>*   **Potential for Price Swings:** There's also an indication of potential price swings and volatility. One analysis suggests the stock could oscillate in the $200 to $210 range in the short term, with a slight upward bias.
>*   **Resistance Levels:** Watch for resistance around $208.64 and $210. A break above these levels could lead to a short-term rally.
>*   **Support Levels:** Key support levels to monitor are around $198.15 and $191.10.
>*   **Mixed Signals:** Some mixed signals exist, including a sell signal from a pivot top point in late June and a sell signal from the 3-month Moving Average Convergence Divergence (MACD). However, buy signals are present from both short and long->term Moving Averages.
>
>**Bullish Predictions:**
>
>*   **Analyst Ratings:** A large majority of analysts have a "Buy" or "Strong Buy" rating on the stock.
>*   **Price Targets:** The consensus average price target varies, but it generally ranges from $243 to $246, suggesting a potential upside. Some analysts have higher price targets, reaching up to $290 or even $305.
>*   **Factors Driving Bullish Sentiment:**
>    *   Growth in Amazon Web Services (AWS) and its expansion in cloud-generated revenue and AI.
>    *   Amazon's investment in and development of AI technologies.
>    *   Continued growth in e-commerce.
>    *   Successful Prime Day sales events.
>*   **Long-Term Potential:** Some bullish scenarios predict a price in the high $400s by 2030.
>
>**Bearish Predictions:**
>
>*   **Downside Risks:**
>    *   Increased competition in e-commerce and cloud computing.
>    *   Regulatory scrutiny.
>    *   Macroeconomic factors.
>    *   Large insider sales (e.g., Jeff Bezos selling shares) can sometimes be interpreted negatively.
>*   **Bearish Price Targets:** Some bearish forecasts suggest the price could fall to around $206 or even lower in the short term. One analysis suggests a potential gain for short sellers in the near term.
>*   **Bearish Long-Term Scenarios:** A bearish long-term case estimates the stock price could be as low as $77 by 2030.
>
>**Important Considerations:**
>
>*   **Analyst estimates are not guarantees:** Forecasts and ratings are subject to change based on market conditions and company performance.
>*   **Do your own research:** It's essential to consider your personal investment goals, risk tolerance, and conduct thorough research before making any investment decisions.


# The (current) limits reached

With two prompts Gemini 2.0 made all the effort I've spent on finance api's obsolete. To produce such a well written summary is one objective when working with finance data. This is great! Now all we need is a generative AI capable in our own language. There's a limit of course. The grounding is subjectively true based only on it's grounding supports -- it may even be hallucinated:

```python
response = chat.send_message('''What is mgm studio's stock ticker symbol?''')
Markdown(response.text)
```

>It's a bit complicated because of the history of MGM. Here's the breakdown:
>
>*   **MGM Resorts International:** This is the publicly traded company focused on hotels, casinos, and entertainment venues. Its ticker symbol is **MGM** and it's listed on the New York Stock Exchange (NYSE).
>*   **Metro-Goldwyn-Mayer (MGM) Studios:** This is the film and television production company. It was acquired by Amazon in 2022 and is now a subsidiary called Amazon MGM Studios. Therefore, there is no separate ticker symbol for MGM Studios. It's >part of Amazon (AMZN).
>*   **MGM Holdings Inc.:** This *was* a publicly traded company (ticker: MGMB) that held Metro-Goldwyn-Mayer (MGM) Studios. However, since Amazon acquired MGM, MGM Holdings Inc. is no longer a separate publicly traded entity.
>
>**In summary:** If you're looking to invest in the broader MGM brand, which includes hotels and casinos, you'd look at **MGM**. If you're interested in MGM Studios, that's now part of **Amazon (AMZN)**.

The order of results and/or content of results is interesting here. The AI is confused about which MGM Studios I'm referring to. On non-thinking variants Gemini may not even mention Amazon. Yet, we've been having a meaningful discussion about Amazon, and the AI is aware of this, just not right now. Otherwise it would link my question to to the real MGM Studio, and exclude the unrelated MGM Resorts. The confusion is linked to the use of the MGM word token. The unrelated MGM stock ticker has now entered the discussion. Depending on how you prompt Gemini 2.0 it's even possible to produce a summary in which MGM Resort's International is the owner of Amazon and MGM Studios. There's two more caveat. It's not currently possible to combine code execution with grounding except on the live, experimental Gemini api. Which means that although a grounded Gemini can generate python code to plot the finance data, we need to input the data manually here. That includes matching a schema or prompting it's output.

```python
response = chat.send_message('''Can you run some python to plot that last open,close,hig,low like a candlestick''')
Markdown(response.text)
```

>I can't directly execute Python code to generate a candlestick plot within this environment. However, I can provide you with a Python code snippet using the `plotly` library that you can run locally to achieve this.
>
>Here's the code, along with explanations:
>
><snipped...>

```python
response = chat.send_message(
    '''What is AMZN open,close,high,low data for the past month? 
    Present the data with multiple columns for display in markdown.''')
Markdown(response.text)
```

>I am unable to retrieve the exact open, close, high, and low data for AMZN over the *past month* with a single query. However, I can provide the data available from the search results, which covers recent trading days. Keep in mind that this is not >the *entire* past month, but it's the most recent information I could gather.
>
>Here's the data in a markdown table:
>
>| Date       | Open    | High    | Low     | Close   |
>|------------|---------|---------|---------|---------|
>| 2025-07-03 | $221.82 | $224.01 | $221.36 | $223.41 |
>| 2025-07-02 | $219.73 | $221.60 | $219.06 | $219.92 |
>| 2025-07-01 | $219.50 | $221.875| $217.93 | $220.46 |
>| 2025-06-30 | $223.52 | $223.82 | $219.12 | $219.39 |
>| 2025-06-27 | $219.92 | $223.30 | $216.74 | $223.30 |
>| 2025-06-26 | $213.12 | $218.035| $212.01 | $217.12 |
>| 2025-06-25 | $214.615| $216.03 | $211.11 | $211.99 |
>| 2025-06-24 | $212.135| $214.34 | $211.045| $212.77 |
>| 2025-06-23 | $209.79 | $210.39 | $207.3101| $208.47 |
>| 2025-06-20 | $214.68 | $214.89 | $208.2709| $209.69 |
>
>**Important Notes:**
>
>*   **Data Limitations:** This is *not* the complete data for the past month. It only includes the dates for which I could retrieve the information.
>*   **Source:** The data is compiled from the search results.
>*   **Verify Data:** Always verify this information with a reputable financial data provider for accuracy before making any investment decisions.

The second caveat is a lack of access to realtime data. Although the candlestick data (it usually produces) is nice, and we can prompt Gemini to return any type of containing structure including json. It also produces non-deterministic output for all stock symbols. Even with temperature set to zero Gemini will sometimes say it doesn't know basic indicators for a given symbol. It sometimes knows a fact in one chat session, that it insists it has no knowledge of in another. Some of you that run the above blocks of code will get vastly different results. Sometimes including the whole month of candlestick data.


# Enter StockChat

Still, with a total of four prompts Gemini replaces all past effort on wrapping finance api's. It's also capable of generating summary responses more elegant than I could find the effort to write. Enter StockChat, the assistant that knows finance data. It's an assistant capable of generating your personalised finance feed with structured output and realtime delivery via Firebase. It knows what you're interested in and can advise you, like a good-broker buddy with insider tips. It has the spreadsheets but knows you don't want to see them. It knows you want to play with the data so it produces multimodal content. 

In order to solve these problems we'll need to move beyond a basic chat session to a multi-tool approach. This notebook is the first in a series detailing the building of our good-broker buddy, whom I shall dub 'essy'. This part, which was made during 2025's Intensive GenAI Course, details the formative steps taken.

The main problem to address before starting is the state of multi-tool support in Gemini-2.0. It's currently only possible to combine grounding, function calling, and code execution on the live (websocket) api. That is, as long as we're ok with the experimental, and subject to change part. Clearly that's not an option for our Essy. We'll start with a multi-model approach. Each expert can be good at different parts of the problem. One such expert will use function calling to chain the models together. One expert to rule them all. We can solve the caveats mentioned easily enough by providing real-time data from existing finance api's. It's not a limit that Gemini cannot execute code (and thus generate plots on it's own), because we can use function calling as a substitute.

We can't have a knowledgeable Essy without a vector database to store our knowledge. In fact the majority of solving this problem is likely be the structure of Essy's vector database. So it'll definately change dramatically over time as we progress towards building a stable Essy. We'll use the popular Chroma and build a RAG expert to begin. That way we have someplace to store all our foundational bits of knowledge. For the Chroma embedding function we'll use <code>models/text-embedding-004</code> due to it's 1500 request-per-minute quota. We'll need to be mindful of the smaller 2,048 token input. Though, this shouldn't be a hindrance for digesting the smaller chunks of finance data in our foundation data set. For the augmented generation phase we'll use <code>models/gemini-2.0-flash</code> variants due to it's 1500 request-per-day quota.


# Instantiate the Tools

Let's load some test data and see what the RAG can do. The test data is a CSV file containing stock market exchange data. It includes the market id code, name, locale, and operating hours. The import will use CSVLoader from <code>langchain-community</code> to parse the exchange data into Documents that our RAG can ingest.

```python
# Instantiate tools and load the exchange data from source csv.
# - Identifies exchanges by a 1-2 letter code which can be used to filter response data.
# - Also maps the exchange code to exchange details.
try:
    df = pandas.read_csv("/kaggle/input/exchanges/exchanges_src.csv")
except FileNotFoundError as e:
    df = pandas.read_csv("exchanges_src.csv") # local run
df = df.drop(["close_date"], axis=1).fillna("")
df.to_csv("exchanges.csv", index=False)
exchanges = CSVLoader(file_path="exchanges.csv", encoding="utf-8", csv_args={"delimiter": ","}).load()

# Prepare a RAG tool for use and add the exchange data.
tool_rag = RetrievalAugmentedGenerator(api.client, "finance")
tool_rag.add_documents_list(exchanges)

# Prepare a the grounding tools for use.
tool_wiki = WikiGroundingGenerator(api.client, tool_rag)
tool_ground = SearchGroundingGenerator(api.client, tool_rag)
tool_rest = RestGroundingGenerator(tool_rag, with_limits=True)
```

>Generate document embedding: 0it [00:00, ?it/s]


Now that the data is loaded lets ask our RAG to perform some augmenting. We can ask it to perform all sorts of useful tasks. We'll generate some useful reusable data structures and check to make sure it can answer important questions. The exchanges all have id's which are used to filter the realtime data. So we'll make sure the RAG know how to create this mapping. We'll also check it's awareness of operating hours. After all, Essy, doesn't mindlessly hammer away at api's when no new data is available.

```python
# The RAG tool is a helpful expert.

response = tool_rag.get_exchanges_csv(
    """Give me a dictionary in string form. It must contain key:value pairs mapping 
    exchange code to name. Just the dictionary string in pretty form.""")
print(response.text)

response = tool_rag.get_exchanges_csv(
    """What is the Germany exchange code? Return only the exchange codes as a simple 
    comma separated value that I can copy.""")
print(response.text, "\n")

response = tool_rag.get_exchanges_csv("What are the Germany exchanges and thier corresponding exchange codes?")
print(response.text, "\n")

response = tool_rag.generate_answer("What are Google's stock ticker symbols?")
print(response.text, "\n")

response = tool_rag.generate_answer("What is Facebook's stock ticker symbol?")
print(response.text, "\n")

response = tool_rag.get_exchanges_csv("What are the US exchange operating hours?")
print(response.text, "\n")

response = tool_rag.get_exchanges_csv(
    f"""Answer based on your knowledge of exchange operating hours.
    Do not answer in full sentences. Omit all chat and provide the answer only.
    The fields pre_market and post_market both represent extended operating hours.

    The current date and time: {datetime.now(GeneratedEvent.tz()).strftime('%c')}

    Weekdays are: Mon, Tue, Wed, Thu, Fri.
    On weekdays all exchanges open after pre-market and regular hours.
    On weekdays all exchanges close after regular and post-market hours.
    
    Weekends are: Sat, Sun.
    Always exclude weekends from exchange operating hours.
    A list of holidays in date format mm-dd-yyyy: {tool_rag.holidays["US"]}
    Always exclude holidays from exchange operating hours.
    When the answer is a holiday use the prior weekday for close.
    When the answer is a holiday use the next weekday for open.
    
    Consider the US exchange's operating hours.
    Provide the most recent weekday's close including post_market hours.
    
    Answer with a date that uses this format: '%a %b %d %X %Y'.""")
print(response.text)
```

>    ```
>    {
>        "SC": "BOERSE_FRANKFURT_ZERTIFIKATE",
>        "SX": "DEUTSCHE BOERSE Stoxx",
>        "HK": "HONG KONG EXCHANGES AND CLEARING LTD",
>        "DB": "DUBAI FINANCIAL MARKET",
>        "NZ": "NEW ZEALAND EXCHANGE LTD",
>        "QA": "QATAR EXCHANGE",
>        "KS": "KOREA EXCHANGE (STOCK MARKET)",
>        "SW": "SWISS EXCHANGE",
>        "DU": "BOERSE DUESSELDORF",
>        "BC": "BOLSA DE VALORES DE COLOMBIA",
>        "KQ": "KOREA EXCHANGE (KOSDAQ)",
>        "SN": "SANTIAGO STOCK EXCHANGE",
>        "SI": "SINGAPORE EXCHANGE",
>        "AD": "ABU DHABI SECURITIES EXCHANGE",
>        "CO": "OMX NORDIC EXCHANGE COPENHAGEN A/S",
>        "L": "LONDON STOCK EXCHANGE",
>        "ME": "MOSCOW EXCHANGE",
>        "TO": "TORONTO STOCK EXCHANGE",
>        "BD": "BUDAPEST STOCK EXCHANGE",
>        "TG": "DEUTSCHE BOERSE TradeGate",
>        "US": "US exchanges (NYSE, Nasdaq)",
>        "TW": "TAIWAN STOCK EXCHANGE",
>        "JK": "INDONESIA STOCK EXCHANGE",
>        "SZ": "SHENZHEN STOCK EXCHANGE",
>        "VS": "NASDAQ OMX VILNIUS",
>        "MX": "BOLSA MEXICANA DE VALORES (MEXICAN STOCK EXCHANGE)",
>        "DE": "XETRA",
>        "PR": "PRAGUE STOCK EXCHANGE",
>        "BK": "STOCK EXCHANGE OF THAILAND",
>        "VI": "Vienna Stock Exchange",
>        "MU": "BOERSE MUENCHEN",
>        "KL": "BURSA MALAYSIA",
>        "BE": "BOERSE BERLIN",
>        "T": "TOKYO STOCK EXCHANGE-TOKYO PRO MARKET",
>        "V": "TSX VENTURE EXCHANGE - NEX",
>        "PA": "NYSE EURONEXT - MARCHE LIBRE PARIS",
>        "PM": "Philippine Stock Exchange",
>        "IR": "IRISH STOCK EXCHANGE - ALL MARKET",
>        "TA": "TEL AVIV STOCK EXCHANGE",
>        "IC": "NASDAQ OMX ICELAND",
>        "SG": "BOERSE STUTTGART",
>        "MC": "BOLSA DE MADRID",
>        "VN": "Vietnam exchanges including HOSE, HNX and UPCOM",
>        "HM": "HANSEATISCHE WERTPAPIERBOERSE HAMBURG",
>        "CR": "CARACAS STOCK EXCHANGE",
>        "SS": "SHANGHAI STOCK EXCHANGE",
>        "BR": "NYSE EURONEXT - EURONEXT BRUSSELS",
>        "IS": "BORSA ISTANBUL",
>        "AX": "ASX - ALL MARKETS",
>        "KW": "Kuwait Stock Exchange",
>        "NE": "AEQUITAS NEO EXCHANGE",
>        "SR": "SAUDI STOCK EXCHANGE",
>        "F": "DEUTSCHE BOERSE AG",
>        "SA": "Brazil Bolsa - Sao Paolo",
>        "CA": "Egyptian Stock Exchange",
>        "MT": "MALTA STOCK EXCHANGE",
>        "AT": "ATHENS EXCHANGE S.A. CASH MARKET",
>        "HA": "Hanover Stock Exchange",
>        "BH": "BAHRAIN BOURSE",
>        "AS": "NYSE EURONEXT - EURONEXT AMSTERDAM",
>        "WA": "WARSAW STOCK EXCHANGE/EQUITIES/MAIN MARKET",
>        "ST": "NASDAQ OMX NORDIC STOCKHOLM",
>        "MI": "Italian Stock Exchange",
>        "LS": "NYSE EURONEXT - EURONEXT LISBON",
>        "JO": "JOHANNESBURG STOCK EXCHANGE",
>        "BA": "BOLSA DE COMERCIO DE BUENOS AIRES",
>        "HE": "NASDAQ OMX HELSINKI LTD",
>        "OL": "OSLO BORS ASA",
>        "TL": "NASDAQ OMX TALLINN",
>        "TWO": "TPEx",
>        "CS": "CASABLANCA STOCK EXCHANGE",
>        "RO": "BUCHAREST STOCK EXCHANGE",
>        "NS": "NATIONAL STOCK EXCHANGE OF INDIA",
>        "BO": "BSE LTD",
>        "RG": "NASDAQ OMX RIGA",
>        "CN": "CANADIAN NATIONAL STOCK EXCHANGE",
>        "NL": "Nigerian Stock Exchange"
>    }
>    ```
>
>    BE, SX, TG, DE, DU, F, MU, SG, SC, HM, HA
>        
>    The Germany exchanges and their corresponding codes are as follows: BOERSE BERLIN (BE), BOERSE DUESSELDORF (DU), XETRA (DE), BOERSE MUENCHEN (MU), DEUTSCHE BOERSE Stoxx (SX), DEUTSCHE BOERSE AG (F), HANSEATISCHE WERTPAPIERBOERSE HAMBURG (HM), >BOERSE STUTTGART (SG), Hanover Stock Exchange (HA), DEUTSCHE BOERSE TradeGate (TG), and BOERSE_FRANKFURT_ZERTIFIKATE (SC).
>     
>    I don't know.
>        
>    I don't know.
>        
>    In the United States, pre-market trading hours are from 04:00 to 09:30, regular trading hours are from 09:30 to 16:00, and post-market trading hours are from 16:00 to 20:00, all in the America/New_York time zone. These hours apply to exchanges >such as NYSE and Nasdaq.
>        
>    Fri Jul 04 20:00:00 2025

Excellent! Though, despite my best effort I could not convince Gemini to apply date correction (during chaining) based on holiday. It simply wasn't stable enough to be useful. I would either have to add a holiday data set, or (what I chose) apply a quick temporary fix. A real-time API endpoint may fail due to a holiday being selected as the date. If that happens I'll just retry Thursday if the failure happened on Friday, likewise choosing Friday if the failure happened on Monday. Crude but simple for foundational purposes.


# Declaring the Function Calling Metadata

Our Function Calling expert will chain together the other experts we've implemented thus far. It also provides the final response through augmentation. This time using the tools as a source of grounding truth. It'd like to say it's all truth organised by topic and other metadata. It's still a precarious situation if Essy incidently chains into mining data on another topic. We want Amazon to be the owner of MGM Studio's not MGM Resorts International. We also don't want a summary to include another company unless that company is a peer.

The function calling metadata is thus extremely important. It needs to combine our other experts with the real-time api's data. Essy will use two API providers as sources of finance data. The primary motivation being that each provider has limits in their own way, yet both are useful in their own own way. This is useful anywhere you need a broad spectrum of sources of truth. At metadata creation I'll adopt the naming convention of appending the provider (if any) id. This helps keep functions more understandable when you know which provider you're dealing with.


# Implementing the Function Calling Expert

One downside of this part being the main part was the lack of time to refactor this part more. Our formative Essy implements as much useful data from two finacial APIs. In order to use it you will need to declare secrets for <a class="anchor-link" href="https://finnhub.io/dashboard">Finnhub</a> and <a class="anchor-link" href="https://polygon.io/dashboard">Polygon</a> finance APIs. Register at their respective sites for your free API key. Then import the secret using the same method as how you setup Google's API key.


# Define the System Prompt

```python
# Define the system prompt.

instruction = f"""You are a helpful and informative bot that answers finance and stock market questions. 
Only answer the question asked and do not change topic. While the answer is still
unknown you must follow these rules for predicting function call order:

RULE#1: Always consult your other functions before get_search_tool_response.
RULE#2: Always consult get_wiki_tool_response before get_search_tool_response.
RULE#3: Always consult get_search_tool_response last.
RULE#4: Always convert timestamps with get_local_datetime and use the converted date/time in your response.
RULE#5: Always incorporate as much useful information from tools and functions in your response."""
```

# Import the Rest API Keys


```python
# Import the finance api secret keys.

POLYGON_API_KEY = UserSecretsClient().get_secret("POLYGON_API_KEY")
FINNHUB_API_KEY = UserSecretsClient().get_secret("FINNHUB_API_KEY")
```

# Ask a question

```python
send_message("What is the current session for US exchanges?")
```

The current market session for US exchanges is closed.

```python
send_message("What is the US market status?")
```

The US market is currently closed. The market session is closed, and it is not a holiday. The status was last updated on Sunday, July 6, 2025, at 8:12:25 PM America/New_York time.

```python
send_message("When was the last US market close?")
```

The last United States market close was on Thursday, July 3, 2025, at 8:00 PM Eastern Time.

```python
send_message("What is Apple's stock ticker?")
```

Apple's stock ticker is AAPL. The current price is \$213.55, with a change of \$1.11 and a percent change of 0.5225%. The high for the day is \$214.65, and the low is \$211.81. The opening price was \$212.145, and the previous close was \$212.44.


```python
send_message("What is the current price of Amazon stock? Display the result as a json object.")
```

The current price of Amazon (AMZN) stock is \$223.41. This is a change of \$3.49, or 1.5869% from the previous close. Today's high was \$224.01, and the low was \$221.36. The opening price was \$221.82. The price was last updated on Saturday, July 5, 2025 at 4:00:00 PM.

```python
send_message("Show me Apple's basic financials. How has the stock performed?")
```

Here's a summary of Apple's basic financials and stock performance:

**Basic Financials (as of March 29, 2025, unless otherwise specified):**

*   **Market Capitalization:** \$3,189,545.8 million
*   **Earnings Per Share (TTM):** \$6.4078
*   **Price-to-Earnings Ratio (TTM):** 32.7826
*   **Revenue Per Share (TTM):** \$26.5916
*   **Price-to-Sales Ratio (TTM):** 7.9666
*   **Book Value Per Share (Quarterly):** \$4.4712
*   **Price-to-Book Ratio (Quarterly):** 50.0084
*   **Cash Flow Per Share (TTM):** \$6.86253
*   **Price-to-Cash Flow Ratio (TTM):** 29.1134
*   **Free Cash Flow Per Share (TTM):** \$6.5924
*   **Price-to-Free Cash Flow Ratio (TTM):** 32.3858
*   **Gross Margin (TTM):** 46.63%
*   **Operating Margin (TTM):** 31.81%
*   **Net Profit Margin (TTM):** 24.3%
*   **Return on Assets (TTM):** 28.37%
*   **Return on Equity (TTM):** 151.31%
*   **Return on Invested Capital (TTM):** 58.95%
*   **Current Ratio (Quarterly):** 0.8209
*   **Quick Ratio (Quarterly):** 0.7775
*   **Long-Term Debt to Equity (Quarterly):** 1.1762
*   **Total Debt to Equity (Quarterly):** 1.4699
*   **52-Week High:** \$260.1 (reached on December 26, 2024)
*   **52-Week Low:** \$169.2101 (reached on April 08, 2025)
*   **Year-to-Date Price Return:** -14.7233%
*   **52-Week Price Return:** -3.0508%

**Stock Performance (as of July 05, 2025, 20:00:00 Eastern Time):**

*   **Current Price:** \$213.55
*   **Change:** \$1.11
*   **Percent Change:** 0.5225%
*   **High Price of the Day:** \$214.65
*   **Low Price of the Day:** \$211.8101
*   **Open Price of the Day:** \$212.145
*   **Previous Close Price:** \$212.44

```python
send_message("I need Apple's daily candlestick from 2025-05-05")
```

On 2025-05-05, Apple's daily candlestick data showed the following:

*   **Open Price:** 203.1
*   **High Price:** 204.1
*   **Low Price:** 198.21
*   **Close Price:** 198.89
*   **Volume:** 69,018,452
*   **Pre-Market Price:** 205.0
*   **After-Hours Price:** 198.6

```python
send_message("Tell me who are Apple's peers?")
```

Apple's peers in the sub-industry include Dell Technologies -C (DELL), Super Micro Computer Inc (SMCI), Hewlett Packard Enterprise (HPE), HP Inc (HPQ), Western Digital Corp (WDC), Netapp Inc (NTAP), Pure Storage Inc - Class A (PSTG), and Ionq Inc (IONQ).

```python
send_message("Tell me the recommendation trends for all of Apple's peers")
```

Here are the recommendation trends for Apple's peers:

**DELL TECHNOLOGIES -C (DELL):**
*   **July 1, 2025:** 7 Strong Buys, 19 Buys, 4 Holds, 0 Sells, 0 Strong Sells
*   **June 1, 2025:** 6 Strong Buys, 20 Buys, 4 Holds, 0 Sells, 0 Strong Sells
*   **May 1, 2025:** 6 Strong Buys, 20 Buys, 4 Holds, 0 Sells, 0 Strong Sells
*   **April 1, 2025:** 6 Strong Buys, 20 Buys, 4 Holds, 0 Sells, 0 Strong Sells

**SUPER MICRO COMPUTER INC (SMCI):**
*   **July 1, 2025:** 2 Strong Buys, 10 Buys, 10 Holds, 2 Sells, 0 Strong Sells
*   **June 1, 2025:** 2 Strong Buys, 10 Buys, 10 Holds, 2 Sells, 0 Strong Sells
*   **May 1, 2025:** 2 Strong Buys, 8 Buys, 9 Holds, 2 Sells, 0 Strong Sells
*   **April 1, 2025:** 2 Strong Buys, 8 Buys, 8 Holds, 2 Sells, 0 Strong Sells

**HEWLETT PACKARD ENTERPRISE (HPE):**
*   **July 1, 2025:** 5 Strong Buys, 7 Buys, 8 Holds, 0 Sells, 0 Strong Sells
*   **June 1, 2025:** 4 Strong Buys, 7 Buys, 8 Holds, 0 Sells, 0 Strong Sells
*   **May 1, 2025:** 4 Strong Buys, 6 Buys, 9 Holds, 0 Sells, 0 Strong Sells
*   **April 1, 2025:** 4 Strong Buys, 7 Buys, 8 Holds, 0 Sells, 0 Strong Sells

**HP INC (HPQ):**
*   **July 1, 2025:** 2 Strong Buys, 4 Buys, 14 Holds, 1 Sell, 0 Strong Sells
*   **June 1, 2025:** 2 Strong Buys, 4 Buys, 13 Holds, 1 Sell, 0 Strong Sells
*   **May 1, 2025:** 2 Strong Buys, 4 Buys, 13 Holds, 1 Sell, 0 Strong Sells
*   **April 1, 2025:** 2 Strong Buys, 4 Buys, 13 Holds, 1 Sell, 0 Strong Sells

**WESTERN DIGITAL CORP (WDC):**
*   **July 1, 2025:** 6 Strong Buys, 19 Buys, 5 Holds, 0 Sells, 0 Strong Sells
*   **June 1, 2025:** 6 Strong Buys, 19 Buys, 5 Holds, 0 Sells, 0 Strong Sells
*   **May 1, 2025:** 6 Strong Buys, 17 Buys, 6 Holds, 0 Sells, 0 Strong Sells
*   **April 1, 2025:** 6 Strong Buys, 15 Buys, 8 Holds, 0 Sells, 0 Strong Sells

**NETAPP INC (NTAP):**
*   **July 1, 2025:** 3 Strong Buys, 9 Buys, 16 Holds, 0 Sells, 0 Strong Sells
*   **June 1, 2025:** 3 Strong Buys, 9 Buys, 16 Holds, 0 Sells, 0 Strong Sells
*   **May 1, 2025:** 3 Strong Buys, 9 Buys, 16 Holds, 0 Sells, 0 Strong Sells
*   **April 1, 2025:** 3 Strong Buys, 8 Buys, 17 Holds, 0 Sells, 0 Strong Sells

**PURE STORAGE INC - CLASS A (PSTG):**
*   **July 1, 2025:** 8 Strong Buys, 12 Buys, 7 Holds, 1 Sell, 0 Strong Sells
*   **June 1, 2025:** 8 Strong Buys, 13 Buys, 6 Holds, 1 Sell, 0 Strong Sells
*   **May 1, 2025:** 8 Strong Buys, 13 Buys, 6 Holds, 1 Sell, 0 Strong Sells
*   **April 1, 2025:** 8 Strong Buys, 13 Buys, 6 Holds, 1 Sell, 0 Strong Sells

**IONQ INC (IONQ):**
*   **July 1, 2025:** 2 Strong Buys, 7 Buys, 2 Holds, 0 Sells, 0 Strong Sells
*   **June 1, 2025:** 2 Strong Buys, 7 Buys, 2 Holds, 0 Sells, 0 Strong Sells
*   **May 1, 2025:** 2 Strong Buys, 7 Buys, 2 Holds, 0 Sells, 0 Strong Sells
*   **April 1, 2025:** 2 Strong Buys, 7 Buys, 2 Holds, 0 Sells, 0 Strong Sells

```python
send_message("Tell me who are Amazon's peers?")
```

Amazon's peers include Coupang Inc (CPNG), eBay Inc (EBAY), Ollie's Bargain Outlet Holdings (OLLI), Dillard's Inc-Class A (DDS), Etsy Inc (ETSY), Macy's Inc (M), Savers Value Village Inc (SVV), and Groupon Inc (GRPN).

```python
api.push_default_model(3) # generate with gemini-2.5-flash
send_message(
    """Tell me Amazon's current share price and provide candlestick data for the past month.
    Sort the data in descending order by date. Format the prices consistently as currency.
    Round prices to two decimal places.
    Present the data with multiple columns for display in markdown.
    Discuss and provide details about any patterns you notice in the price data.
    Correlate recent patterns with news over the same date range.""")
api.pop_default_model()
```

Amazon's current share price is \$223.41, with a daily change of \$3.49 and a percentage change of 1.59%. The high price for the day was \$224.01, the low price was \$221.36, and the open price was \$221.82. The previous close price was \$219.92. The price was recorded on July 3, 2025, at 4:00:00 PM Eastern Time.

Here is the candlestick data for Amazon (AMZN) for the past month, sorted in descending order by date:

| Date (Eastern Time) | Open Price | High Price | Low Price | Close Price | Volume |
|---|---|---|---|---|---|
| July 3, 2025 | \$221.82 | \$224.01 | \$221.36 | \$223.41 | 29,632,353 |
| July 2, 2025 | \$219.92 | \$223.30 | \$216.74 | \$223.30 | 119,217,138 |
| July 1, 2025 | \$213.12 | \$218.04 | \$212.01 | \$217.12 | 50,480,814 |
| June 30, 2025 | \$219.73 | \$221.60 | \$219.06 | \$219.92 | 30,894,178 |
| June 29, 2025 | \$219.50 | \$221.88 | \$217.93 | \$220.46 | 39,256,830 |
| June 28, 2025 | \$212.14 | \$214.34 | \$211.05 | \$212.77 | 38,378,757 |
| June 27, 2025 | \$214.62 | \$216.03 | \$211.11 | \$211.99 | 31,755,698 |
| June 26, 2025 | \$215.19 | \$217.41 | \$214.56 | \$214.82 | 32,086,262 |
| June 25, 2025 | \$209.79 | \$210.39 | \$207.31 | \$208.47 | 37,311,725 |
| June 24, 2025 | \$214.68 | \$214.89 | \$208.27 | \$209.69 | 75,350,733 |
| June 23, 2025 | \$215.09 | \$217.96 | \$212.34 | \$212.52 | 44,360,509 |
| June 20, 2025 | \$211.78 | \$213.58 | \$211.33 | \$213.24 | 27,639,991 |
| June 19, 2025 | \$214.75 | \$217.85 | \$212.88 | \$216.98 | 38,102,502 |
| June 18, 2025 | \$206.55 | \$208.18 | \$205.18 | \$207.23 | 29,915,592 |
| June 17, 2025 | \$212.31 | \$217.06 | \$211.60 | \$216.10 | 33,284,158 |
| June 16, 2025 | \$209.96 | \$214.05 | \$209.62 | \$212.10 | 29,337,763 |
| June 13, 2025 | \$207.11 | \$208.95 | \$205.03 | \$205.71 | 33,139,121 |
| June 12, 2025 | \$209.55 | \$212.81 | \$207.56 | \$207.91 | 51,979,243 |
| June 11, 2025 | \$223.52 | \$223.82 | \$219.12 | \$219.39 | 58,887,780 |
| June 10, 2025 | \$217.41 | \$218.40 | \$212.89 | \$213.20 | 39,325,981 |
| June 9, 2025 | \$216.78 | \$217.69 | \$214.15 | \$217.61 | 31,303,317 |
| June 6, 2025 | \$212.40 | \$213.87 | \$210.50 | \$213.57 | 39,832,500 |

**Price Patterns and Correlation with News:**

Looking at the candlestick data, Amazon's stock price has generally been on an upward trend over the past month, with some fluctuations. There's a notable dip around June 13th, followed by a recovery and a more significant increase towards the end of June and early July.

Correlating these patterns with the news, several themes emerge:

*   **Artificial Intelligence (AI) and Cloud Computing (AWS) Optimism:** A significant portion of the news articles highlights Amazon's strong position and continued investment in AI and its Amazon Web Services (AWS) cloud computing division. Many articles, particularly from mid-June onwards, emphasize AWS as a key growth driver and a major beneficiary of the AI boom. This positive sentiment around AI and AWS likely contributed to the stock's upward movement. For example, articles on June 13th, 17th, 20th, 22nd, 24th, 25th, 26th, and July 3rd all discuss Amazon's AI and AWS strengths, with many analysts predicting significant future growth.
*   **E-commerce and Prime Day:** While AI and AWS are prominent, Amazon's core e-commerce business and the upcoming Prime Day (mentioned for July 8-11, 2025) also appear in the news. Articles on June 23rd and July 3rd discuss Amazon leveraging luxury beauty products for Prime Day to offset tariff impacts, and MiLi preparing for Prime Day deals. The anticipation of strong Prime Day sales could have contributed to positive investor sentiment.
*   **Strategic Partnerships and Diversification:** News about Amazon's strategic partnerships, such as with Roku for advertising (June 28th), and its expansion into areas like satellite internet (Project Kuiper, June 16th) and autonomous vehicles (Zoox, June 25th), demonstrate the company's diversification and potential for future growth beyond its traditional segments. These announcements likely reinforced investor confidence.
*   **Analyst Sentiment and Investor Interest:** Several articles mention positive analyst ratings and price targets for Amazon, with some even suggesting it could surpass other tech giants in market capitalization. The fact that prominent investors like Warren Buffett, Cathie Wood, and Bill Ackman hold Amazon stock (June 13th, 14th, 22nd, 26th, June 4th) further indicates strong investor interest and belief in the company's long-term prospects.
*   **Workforce and Automation:** News on June 18th and 24th discusses Amazon's CEO Andy Jassy stating that AI and automation will lead to a reduction in the workforce. While this might seem negative, the context in the articles often frames it as a strategic move to improve efficiency and profitability, which could be viewed positively by investors.

In summary, the general upward trend in Amazon's stock price over the past month appears to be strongly correlated with positive news surrounding its leadership in artificial intelligence and cloud computing, strategic business expansions, and continued strong analyst and investor confidence. While there are minor fluctuations, the overarching narrative from the news suggests a company well-positioned for future growth.

```python
send_message("What is Apple's ticker overview")
```

Apple Inc. (ticker: AAPL) is a prominent company listed on the NASDAQ Stock Market (XNAS). It boasts a substantial market capitalization of \$3,064,383,420,420.00.

The company's headquarters are located at One Apple Park Way, Cupertino, CA 95014, and their phone number is (408) 996-1010. Their official website is https://www.apple.com.

Apple is recognized as one of the world's largest companies, offering a diverse range of hardware and software products to both consumers and businesses. The iPhone accounts for a significant portion of its sales, with other products like Mac, iPad, and Watch designed to integrate seamlessly within the iPhone-centric software ecosystem. Apple has been actively expanding its services to include new applications such as streaming video, subscription bundles, and augmented reality. The company designs its own software and semiconductors, relying on subcontractors like Foxconn and TSMC for manufacturing. Approximately half of Apple's sales are direct through its flagship stores, while the majority are generated indirectly through partnerships and distribution channels.

Apple operates within the Electronic Computers industry, as indicated by its Standard Industrial Classification (SIC) code 3571. The company was first listed on December 12, 1980, and currently employs a total of 164,000 individuals.

```python
send_message("Tell me about Amazon's historical and current recommendation trends")
```

Here is a summary of Amazon's (AMZN) historical and current recommendation trends:

**Current Recommendation Trends (July 2025):**
As of July 1, 2025, Amazon has received overwhelmingly positive recommendations from analysts. There are 50 'Buy' recommendations, 24 'Strong Buy' recommendations, 5 'Hold' recommendations, and no 'Sell' or 'Strong Sell' recommendations.

**Historical Recommendation Trends (April - June 2025):**
The recommendation trends for Amazon have remained consistently positive over the past few months.

*   **June 2025:** The recommendations were identical to July 2025, with 50 'Buy', 24 'Strong Buy', 5 'Hold', and no 'Sell' or 'Strong Sell' recommendations.
*   **May 2025:** There were 51 'Buy' recommendations, 22 'Strong Buy' recommendations, 6 'Hold' recommendations, and no 'Sell' or 'Strong Sell' recommendations.
*   **April 2025:** Amazon received 50 'Buy' recommendations, 23 'Strong Buy' recommendations, 4 'Hold' recommendations, and no 'Sell' or 'Strong Sell' recommendations.

Overall, the data indicates a strong and consistent positive sentiment from analysts towards Amazon, with a high number of 'Buy' and 'Strong Buy' recommendations and very few 'Hold' or 'Sell' recommendations over the last few months.

```python
send_message("What is Google's stock ticker symbol?")
```

Google's stock ticker symbols are GOOGL and GOOG. These symbols now refer to Alphabet Incorporated, Google's parent company, since the fourth quarter of 2015. Google initially became a public company on August 19, 2004, opening on the NASDAQ National Market under the ticker symbol GOOGL.

```python
send_message("What is MGM Studio's stock symbol?")
```

MGM Studio, which is now known as Amazon MGM Studios after its acquisition by Amazon in 2022 and subsequent merger with Amazon Studios in October 2023, does not have its own stock symbol.

```python
send_message("What is MGM Studio's owner company stock symbol?")
```

MGM Studio has been owned by Amazon since 2022. Amazon's stock symbol is AMZN.

```python
send_message("What is Facebook's stock ticker symbol?")
```

Facebook's stock ticker symbol is META. The company changed its ticker symbol from FB to META on June 9, 2022, as part of its rebranding to Meta Platforms, Inc.

```python
send_message(
    '''Tell me about Amazon's current bullish versus bearish predictions, and recommendation trends.
    Include a discussion of any short-term trends, and sentiment analysis.''')
```

Amazon (AMZN) is currently experiencing a predominantly bullish outlook from analysts, with strong positive sentiment reflected in recent recommendation trends and news coverage.

**Recommendation Trends:**
As of July 2025, analyst recommendations for Amazon are overwhelmingly positive. Out of the latest recommendations, 50 are "Buy" and 24 are "Strong Buy," with only 5 "Hold" recommendations and no "Sell" or "Strong Sell" ratings. This consistent trend of strong buy recommendations has been observed over the past few months, indicating sustained confidence in the company's performance and future prospects.

**Short-Term Trends and Sentiment Analysis:**
Recent news and sentiment analysis highlight several key short-term trends for Amazon:

*   **Artificial Intelligence (AI) and Cloud Computing (Amazon Web Services - AWS):** Amazon's significant investments and advancements in AI are a major driver of positive sentiment. AWS is consistently cited as a leading cloud computing provider, benefiting from the surge in AI spending and providing crucial infrastructure for AI development. Amazon is also developing its own custom AI chips, such as Trainium and Inferentia, and integrating AI across its e-commerce and logistics operations to enhance efficiency and reduce costs. Analysts project AWS alone could be valued at over \$3 trillion.
*   **E-commerce and Advertising Growth:** Amazon maintains a dominant position in the e-commerce market, with a substantial market share in the U.S. The upcoming Prime Day event (July 8-11, 2025) is anticipated to generate significant sales. The company's advertising business is also experiencing robust growth, with strategic partnerships, such as with Roku, expanding its reach and contributing significantly to profitability.
*   **Robotics and Automation:** Amazon is actively deploying robotics in its fulfillment centers and developing next-generation AI-powered robots. This focus on automation is expected to drive productivity gains and improve profit margins, although it may lead to a reduction in the overall workforce, as noted by CEO Andy Jassy.
*   **Diversified Business Model:** Amazon's diversified revenue streams across e-commerce, cloud services, advertising, and streaming are seen as a key strength, providing resilience and multiple avenues for continued growth.
*   **Investor Confidence:** Prominent investors, including Bill Ackman and Warren Buffett, have either increased or maintained significant stakes in Amazon, viewing it as a compelling long-term investment due to its AI-driven growth and strong market positions. The stock has shown positive movements in the S&P 500, with some analysts predicting a potential breakout.

While there are minor neutral or slightly negative sentiments, such as the competitive landscape in various sectors, the impact of tariffs (which Amazon is strategically offsetting with high-margin products), and the noted workforce reduction due to automation, these are largely overshadowed by the overwhelming positive outlook on Amazon's strategic direction, technological innovation, and strong market fundamentals. The overall sentiment indicates a strong belief in Amazon's continued market outperformance and its pivotal role in the evolving AI landscape.

```python
send_message(
    '''Tell me about Google's share price from May 01 2025 until today in a markdown table.
    How has the stock performed?
    Perform a sentiment analysis of news during the same dates. Include trends in your analysis.''')
```

Here's a concise and detailed summary of Google's (Alphabet Inc.) share price performance and a sentiment analysis of related news from May 1, 2025, until July 7, 2025.

**Google (Alphabet Inc.) Share Price Performance (May 1, 2025 - July 3, 2025)**

| Date (Eastern Time) | Open Price | High Price | Low Price | Close Price | Volume |
| :------------------ | :--------- | :--------- | :-------- | :---------- | :----- |
| Thursday, May 1, 2025 | \$160.45 | \$161.95 | \$158.91 | \$161.30 | 30,203,248 |
| Friday, May 2, 2025 | \$163.40 | \$164.97 | \$161.87 | \$164.03 | 25,715,005 |
| Monday, May 5, 2025 | \$163.00 | \$165.39 | \$162.72 | \$164.21 | 21,341,814 |
| Tuesday, May 6, 2025 | \$162.17 | \$164.80 | \$161.19 | \$163.23 | 21,277,210 |
| Wednesday, May 7, 2025 | \$164.08 | \$165.00 | \$147.84 | \$151.38 | 127,747,554 |
| Thursday, May 8, 2025 | \$155.00 | \$155.93 | \$152.90 | \$154.28 | 57,498,692 |
| Friday, May 9, 2025 | \$154.17 | \$155.05 | \$152.20 | \$152.75 | 32,435,281 |
| Monday, May 12, 2025 | \$157.48 | \$159.10 | \$156.25 | \$158.46 | 44,138,818 |
| Tuesday, May 13, 2025 | \$158.79 | \$160.57 | \$156.16 | \$159.53 | 42,382,126 |
| Wednesday, May 14, 2025 | \$159.96 | \$167.00 | \$159.61 | \$165.37 | 48,755,869 |
| Thursday, May 15, 2025 | \$165.84 | \$166.20 | \$162.37 | \$163.96 | 33,146,669 |
| Friday, May 16, 2025 | \$167.72 | \$169.35 | \$165.62 | \$166.19 | 42,846,925 |
| Monday, May 19, 2025 | \$164.51 | \$166.64 | \$164.22 | \$166.54 | 30,426,097 |
| Tuesday, May 20, 2025 | \$166.43 | \$168.50 | \$162.90 | \$163.98 | 46,607,656 |
| Wednesday, May 21, 2025 | \$163.69 | \$173.14 | \$163.56 | \$168.56 | 73,415,956 |
| Thursday, May 22, 2025 | \$171.85 | \$176.77 | \$170.71 | \$170.87 | 74,864,418 |
| Friday, May 23, 2025 | \$169.05 | \$169.96 | \$167.89 | \$168.47 | 35,211,439 |
| Tuesday, May 27, 2025 | \$170.16 | \$173.17 | \$170.00 | \$172.90 | 37,995,670 |
| Wednesday, May 28, 2025 | \$173.16 | \$175.26 | \$171.91 | \$172.36 | 34,783,997 |
| Thursday, May 29, 2025 | \$174.00 | \$174.41 | \$170.63 | \$171.86 | 29,373,803 |
| Friday, May 30, 2025 | \$171.35 | \$172.20 | \$167.44 | \$171.74 | 52,639,911 |
| Monday, June 2, 2025 | \$167.84 | \$169.87 | \$167.39 | \$169.03 | 38,612,272 |
| Tuesday, June 3, 2025 | \$167.49 | \$168.47 | \$165.28 | \$166.18 | 45,084,903 |
| Wednesday, June 4, 2025 | \$166.73 | \$168.21 | \$166.36 | \$168.05 | 26,900,838 |
| Thursday, June 5, 2025 | \$170.34 | \$170.93 | \$167.59 | \$168.21 | 36,444,564 |
| Friday, June 6, 2025 | \$170.83 | \$174.50 | \$170.83 | \$173.68 | 35,731,832 |
| Monday, June 9, 2025 | \$174.54 | \$176.47 | \$174.37 | \$176.09 | 28,935,906 |
| Tuesday, June 10, 2025 | \$176.20 | \$181.10 | \$174.91 | \$178.60 | 61,766,121 |
| Wednesday, June 11, 2025 | \$179.77 | \$180.37 | \$176.75 | \$177.35 | 31,646,757 |
| Thursday, June 12, 2025 | \$176.18 | \$176.72 | \$174.74 | \$175.70 | 20,941,873 |
| Friday, June 13, 2025 | \$172.44 | \$177.13 | \$172.38 | \$174.67 | 27,663,107 |
| Monday, June 16, 2025 | \$174.73 | \$176.94 | \$174.65 | \$176.77 | 27,389,208 |
| Tuesday, June 17, 2025 | \$175.70 | \$177.36 | \$174.58 | \$175.95 | 24,973,043 |
| Wednesday, June 18, 2025 | \$176.01 | \$176.56 | \$173.20 | \$173.32 | 28,707,524 |
| Friday, June 20, 2025 | \$173.94 | \$174.34 | \$165.46 | \$166.64 | 75,659,917 |
| Monday, June 23, 2025 | \$166.27 | \$167.34 | \$162.00 | \$165.19 | 57,670,985 |
| Tuesday, June 24, 2025 | \$166.92 | \$168.22 | \$166.13 | \$166.77 | 40,524,312 |
| Wednesday, June 25, 2025 | \$167.63 | \$172.36 | \$167.55 | \$170.68 | 35,478,989 |
| Thursday, June 26, 2025 | \$172.43 | \$173.69 | \$169.94 | \$173.54 | 31,796,690 |
| Friday, June 27, 2025 | \$173.54 | \$178.68 | \$171.73 | \$178.53 | 108,140,200 |
| Monday, June 30, 2025 | \$180.78 | \$181.23 | \$174.58 | \$176.23 | 63,378,856 |
| Tuesday, July 1, 2025 | \$175.73 | \$176.09 | \$173.53 | \$175.84 | 35,904,526 |
| Wednesday, July 2, 2025 | \$175.54 | \$178.86 | \$175.07 | \$178.64 | 29,128,947 |
| Thursday, July 3, 2025 | \$178.50 | \$179.67 | \$177.05 | \$179.53 | 21,689,729 |

Google's (Alphabet Inc.) stock, traded under the ticker symbol GOOGL, has shown an overall positive performance from May 1, 2025, to July 3, 2025. Starting at a closing price of \$161.30 on May 1st, the stock experienced some fluctuations, including a notable dip to \$151.38 on May 7th. However, it generally trended upwards, closing at \$179.53 on July 3rd. This represents a gain of approximately \$18.23 per share during this period.

**Sentiment Analysis of News and Trends (May 1, 2025 - July 7, 2025)**

A sentiment analysis of news articles related to Google (Alphabet Inc.) during this period reveals a predominantly positive outlook, with some negative and neutral sentiments.

**Overall Sentiment Breakdown:**
*   **Positive:** 126 articles
*   **Negative:** 12 articles
*   **Neutral:** 59 articles

**Key Trends in Sentiment:**

*   **Strong Positive Sentiment on AI and Cloud Computing:** A significant number of articles highlight Google's advancements and leadership in Artificial Intelligence (AI) and its cloud computing services (Google Cloud Platform). Many sources praise Google's AI capabilities, including its Gemini models, AI-powered search, and investments in AI infrastructure. The company is frequently cited as a "no-brainer AI stock to buy" and a leader in the AI race. Its strategic investments in AI, such as scaling AI budgets and developing custom AI chips, are consistently viewed as positive drivers for future growth.
*   **Positive Analyst and Investor Confidence:** Several articles indicate strong confidence from Wall Street analysts and institutional investors in Alphabet's long-term prospects. The stock is often described as "undervalued" and a "bargain" compared to its peers, with predictions of significant stock price appreciation. The company's recent dividend initiation and aggressive share buyback programs are also seen as positive signals for investors.
*   **Waymo's Progress in Autonomous Vehicles:** Google's autonomous vehicle subsidiary, Waymo, is frequently mentioned with positive sentiment. Its progress in robotaxi services, partnerships (e.g., with Uber), and expansion into new markets are highlighted as significant growth drivers for Alphabet.
*   **Concerns and Competition (Negative Sentiment):**
    *   **Impact of AI on Search:** A recurring negative theme is the potential disruption of Google's core search business by new AI-powered search summaries and chatbots. Some analysts express concerns that this could reduce the need for users to visit traditional search engines, potentially impacting Google's advertising revenue.
    *   **Competition in AI and Cloud:** While generally positive, some articles acknowledge intense competition in the AI and cloud computing markets from rivals like Microsoft and Amazon. There are mentions of Google facing challenges in gaining market share in certain cloud segments or from custom silicon solutions developed by competitors.
    *   **Water Usage by Data Centers:** A few articles raise environmental concerns regarding the massive water consumption by data centers, including those operated by Google, which is viewed as a negative impact on the livestock industry.
    *   **Stock Price Fluctuations and External Factors:** Some negative sentiment is tied to specific stock price drops attributed to external factors, such as comments from Apple executives regarding their search relationship or broader market "money flows."

**Analyst Recommendation Trends:**

Analyst recommendations for GOOGL remained consistently positive throughout the period:
*   **July 1, 2025:** 18 Strong Buy, 32 Buy, 13 Hold, 0 Sell, 0 Strong Sell
*   **June 1, 2025:** 18 Strong Buy, 33 Buy, 13 Hold, 0 Sell, 0 Strong Sell
*   **May 1, 2025:** 17 Strong Buy, 33 Buy, 14 Hold, 0 Sell, 0 Strong Sell

This indicates a strong consensus among analysts for a "Buy" or "Strong Buy" rating, reflecting a positive long-term outlook for Alphabet Inc.

In summary, Google's stock has performed positively from May to July 2025, with a general upward trend in its share price. The news sentiment is overwhelmingly positive, driven by the company's strong position and continuous innovation in AI, its growing cloud computing business, and the progress of its autonomous vehicle division, Waymo. While there are some concerns regarding AI's potential impact on traditional search and market competition, analysts maintain a highly optimistic view of Alphabet's future.

```python
send_message(
    '''How is the outlook for Apple based on trends and news sentiment from May 01 2025 until today?
    Perform the same analysis on all peers by sub-industry. Then compare Apple result to it's peers.''')
```

The outlook for Apple (AAPL) from May 1, 2025, to July 7, 2025, has been mixed, with significant news surrounding potential tariffs, AI advancements, and market performance.

**Key Trends and News Sentiment:**

*   **Tariff Concerns:** Apple has faced considerable uncertainty due to potential tariffs imposed by the U.S. government on products manufactured overseas. Several articles highlighted the potential financial impact of these tariffs, with some suggesting a negative sentiment due to increased costs and the possibility of price hikes. However, news of a U.S.-China tariff truce and a pause on tariff increases provided a positive boost, easing concerns about the impact on Apple's supply chain and earnings.
*   **AI and Innovation:** There's a recurring theme of Apple's AI capabilities, particularly with Siri, being perceived as lagging behind competitors. While Apple is investing in AI and exploring partnerships, some analysts and experts expressed disappointment with the lack of revolutionary AI features announced at its WWDC event. Conversely, some articles suggest Apple's focus on user experience and its ecosystem could still drive growth, with potential for AI to enhance its offerings in the future. The development of an AI-powered tabletop robot and the integration of AI technologies from companies like Anthropic and OpenAI are also noted.
*   **Market Performance and Analyst Sentiment:** Apple's stock performance has been described as mixed, with some articles noting underperformance relative to the S&P 500, while others suggest potential buying opportunities during dips. Analysts have expressed varied opinions, with some maintaining positive outlooks based on Apple's strong financials, ecosystem, and potential for future growth, while others caution about valuation and competition. Warren Buffett's Berkshire Hathaway has maintained a significant, though at times reduced, position in Apple, indicating a long-term belief in the company's fundamentals.
*   **Legal and Regulatory Issues:** Apple has faced some legal challenges, including a class-action lawsuit alleging misleading statements about AI features and Siri's capabilities, and a lawsuit against a former engineer for allegedly stealing Vision Pro secrets.

**Peer Analysis (Sub-Industry):**

Apple's peers in the same sub-industry include Dell Technologies (DELL), Super Micro Computer (SMCI), Hewlett Packard Enterprise (HPE), HP Inc. (HPQ), Western Digital (WDC), NetApp (NTAP), Pure Storage (PSTG), and IonQ (IONQ).

*   **Dell Technologies (DELL), Super Micro Computer (SMCI), Hewlett Packard Enterprise (HPE), and HP Inc. (HPQ):** These companies are primarily involved in computing and IT infrastructure. News sentiment for these companies during the period appears to be generally positive, with mentions of strong earnings, strategic partnerships, and growth in areas like AI infrastructure.
*   **Western Digital (WDC) and NetApp (NTAP):** These companies are focused on data storage solutions. Their performance and outlook are often tied to the demand for data storage, which is generally strong due to the increasing amount of data being generated.
*   **Pure Storage (PSTG):** This company specializes in data storage solutions, and its performance is also linked to the broader data storage market trends.
*   **IonQ Inc. (IONQ):** This company is in the quantum computing space, which is a more nascent but high-growth area. Its performance is driven by advancements in quantum technology and its adoption by various industries.

**Comparison of Apple to its Peers:**

*   **AI Focus:** While Apple is investing in AI, its progress and communication around AI have been met with more skepticism compared to some of its peers who are seen as leaders in AI hardware and infrastructure (e.g., Nvidia, which is often mentioned in relation to AI advancements). However, Apple's integrated ecosystem and vast user base provide a unique platform for AI integration.
*   **Market Sentiment:** Apple's outlook has been more heavily influenced by macroeconomic factors like tariffs and regulatory scrutiny, leading to a more mixed sentiment. Some of its peers, particularly those in the hardware and infrastructure space, have seen more consistently positive sentiment driven by strong demand for their products and services, especially in the context of AI development.
*   **Diversification:** Apple's business is highly diversified across hardware, software, and services. Some of its peers are more specialized, focusing on specific areas like data storage or quantum computing, which can lead to different risk and reward profiles.

In summary, while Apple faces challenges related to AI perception and regulatory headwinds, its strong ecosystem and market position remain key strengths. Its peers in the computing and IT infrastructure sectors appear to be benefiting from strong demand, particularly driven by AI, which has led to a more consistently positive sentiment for some of them.

```python
send_message(
    '''What does the recent news say about Apple and the impact of tariffs? From 2025-03-01 up to today.
    Also locate candlestick data for the same dates. 
    Discuss in detail any correlations in patterns between the candlestick and news data.
    Ignore duplicate news entry.''')
```

The recent news regarding Apple and tariffs from March 1, 2025, to July 7, 2025, indicates a significant impact on the company's stock and operations. Several articles highlight President Trump's threats of tariffs on iPhones and other Apple products manufactured overseas, with potential tariff rates ranging from 25% to 104%. This has led to concerns about increased costs for Apple, with estimates suggesting potential impacts of up to \$900 million in additional costs per quarter and a possible 90%+ increase in iPhone costs if tariffs are enforced.

In response to these tariff threats, Apple has been exploring strategies to mitigate the impact, including shifting production to countries like India and Vietnam, and potentially moving some manufacturing to the United States. However, analysts suggest that moving iPhone production to the U.S. is not feasible due to high costs and complex supply chain logistics.

The news also indicates that Apple's stock has experienced volatility due to these tariff concerns, with several articles reporting stock price declines following tariff announcements or threats. Some reports suggest that Apple's stock has underperformed the broader market due to these headwinds.

However, there have also been periods of relief and positive sentiment. Temporary exemptions from tariffs on certain electronics, including smartphones, have provided some relief to Apple. Additionally, pauses in trade talks between the U.S. and China have been viewed as positive developments for Apple, signaling a de-escalation in trade tensions and potentially boosting the stock.

Regarding candlestick data for Apple (AAPL) from March 1, 2025, to July 7, 2025:

*   **March 2025:** The stock experienced a significant downturn in early March, with a notable drop from around \$235 to \$209 by mid-March. There was a slight recovery towards the end of the month, closing around \$213.
*   **April 2025:** April saw considerable volatility. The stock experienced a sharp decline in early April, dropping from around \$210 to a low of \$169 by mid-April. It then showed a strong recovery, climbing to \$198 by the end of the month.
*   **May 2025:** May started with a recovery, with the stock reaching highs around \$223. However, it experienced a decline throughout the month, closing around \$200.
*   **June 2025:** June saw a general upward trend, with the stock starting around \$200 and climbing to a high of \$213 by the end of the month.
*   **Early July 2025:** The stock continued its upward trend in early July, reaching highs around \$208 before a slight pullback.

**Correlations between news and candlestick data:**

The news regarding tariffs and Apple's stock performance shows a clear correlation. Periods of heightened tariff threats and negative news (e.g., early April 2025) coincided with significant drops in Apple's stock price. Conversely, news of tariff pauses, trade deal progress, or temporary exemptions (e.g., mid-April 2025, mid-May 2025) often correlated with stock price recoveries. The overall trend in the candlestick data from March to July 2025 shows significant volatility, with periods of sharp decline followed by recoveries, reflecting the market's reaction to the ongoing tariff developments and Apple's strategic responses. The news suggests that the market is highly sensitive to tariff-related news, which directly impacts investor sentiment and Apple's stock performance.

# Conclusion

For now that will have to do. Our Essy has a solid foundation but more could be done to organise metadata. No evaluation or validation has been performed (except fuzzing the prompt). Next steps include restructuring the vector database based on lessons learned. That'll be followed by plotting, multi-modal, and structured output. The last close date (generative) function can be temperamental. In the same way Gemini always feels regarding dates. I've learnt so much. I'm happy I decided to participate in the event! It really has been a joy to see Essy grow from random chat with Gemini into the foundation for a good-broker buddy. I hope you enjoy playing with this edition as much as I enjoyed building it!

# Update June 7, 2025

Bugfix version 102 finally brings Essy to a stable milestone. A month and a half late :) There's still more to be built including adding reasoning, agents, and structured output. A few unimplemented rest endpoints remain that could make Essy more self-reliant. The vector store has gotten bigger but not smarter. Essy can tell us pre-scored news has some sentiment but cannot generate it due to limited summaries. Essy can detect interesting patterns in a dataset but not between adjacent datasets. There's so much data we'll need to recruit Essy some help.

# Advanced (localhost required)

The functions demonstrated here require a locally running notebook. A dedicated GPU with at least 8GB VRAM is recommended but not required. Output is generated with Gemma 3 12B QAT, Gemma.cpp, and (later) Gemma 3n. Output on Kaggle is based on cached data.

```python
# soon
```
