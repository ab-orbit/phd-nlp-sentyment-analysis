# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class RestaurantReviewsItem(scrapy.Item):
    # define the fields for your item here like:
    restaurant_name = scrapy.Field()
    review_text = scrapy.Field()
    rating = scrapy.Field()
