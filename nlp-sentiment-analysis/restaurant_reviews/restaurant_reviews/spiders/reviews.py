import scrapy
import pandas as pd
import urllib.parse
from restaurant_reviews.items import RestaurantReviewsItem

class ReviewsSpider(scrapy.Spider):
    name = "reviews"
    allowed_domains = ["google.com", "tripadvisor.com", "tripadvisor.com.br"]
    custom_settings = {
        'ROBOTSTXT_OBEY': False
    }

    def start_requests(self):
        csv_path = '/Users/jwcunha/Documents/repos/phd-datascience/nlp-sentiment-analysis/docs/restaurantes_ativos_recife.csv'
        try:
            df = pd.read_csv(csv_path, encoding='latin-1')
        except FileNotFoundError:
            self.logger.error(f"Could not find the CSV file at {csv_path}")
            return

        # Filter restaurants with valid 'nome_fantasia'
        # A valid name has at least 3 alphabetic characters and is not a placeholder
        df = df[df['nome_fantasia'].str.contains('[a-zA-Z]{3,}', na=False) & ~df['nome_fantasia'].str.contains(r'^\*+$', na=False)]

        if df.empty:
            self.logger.info("No restaurants with valid 'nome_fantasia' found in the CSV.")
            return

        for index, row in df.iterrows():
            restaurant_name = row['nome_fantasia']
            search_query = f'"{restaurant_name}" TripAdvisor Recife'
            google_search_url = 'https://www.google.com/search?q=' + urllib.parse.quote(search_query)

            yield scrapy.Request(url=google_search_url, callback=self.parse_google_search, meta={'restaurant_name': restaurant_name})

    def parse_google_search(self, response):
        restaurant_name = response.meta['restaurant_name']

        # Find the first TripAdvisor link in the search results
        # This selector might need to be adjusted based on Google's search result structure
        tripadvisor_link = response.css('a[href*="tripadvisor.com.br/Restaurant_Review"]::attr(href)').get()

        if tripadvisor_link:
            # If the link is a relative URL, it needs to be joined with the base URL
            full_url = response.urljoin(tripadvisor_link)
            yield scrapy.Request(url=full_url, callback=self.parse_restaurant_reviews, meta={'restaurant_name': restaurant_name})
        else:
            self.logger.info(f"No TripAdvisor review page found for {restaurant_name}")

    def parse_restaurant_reviews(self, response):
        restaurant_name = response.meta['restaurant_name']
        self.logger.info(f"Scraping reviews for {restaurant_name} from {response.url}")

        # NOTE: The CSS selectors below are based on a general structure of TripAdvisor pages
        # and might need to be adjusted if the layout has changed.

        # Selector for the review sections
        reviews = response.css('div.review-container')
        if not reviews:
            # Fallback to another possible selector
            reviews = response.css('div.DqMbZ')


        for review in reviews:
            item = RestaurantReviewsItem()
            item['restaurant_name'] = restaurant_name

            # Selector for the review text. It might be split into multiple parts.
            review_text_parts = review.css('span.partial_entry::text').getall()
            if not review_text_parts:
                # Fallback selector
                review_text_parts = review.css('span.JbGkU ::text').getall()

            item['review_text'] = ' '.join(part.strip() for part in review_text_parts if part.strip())

            # Selector for the rating. The rating is often in the class name of a span.
            rating_class = review.css('span.ui_bubble_rating::attr(class)').get()
            if rating_class:
                # Extracts the number from a class like 'ui_bubble_rating bubble_40'
                rating_value = rating_class.split('_')[-1]
                # Convert to a scale of 5
                item['rating'] = str(int(rating_value) / 10)
            else:
                item['rating'] = None


            if item['review_text'] and item['rating']:
                yield item
