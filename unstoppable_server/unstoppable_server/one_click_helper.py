import urllib3
from bs4 import BeautifulSoup
from urllib.parse import quote


def get_one_click_url(food_name):
    http = urllib3.PoolManager()
    url = "http://101.132.162.63/food/search?keyword=" + food_name
    # print(url)
    # print(quote(url, safe='&/:?='))
    html = http.request("GET", quote(url, safe='&/:?='))
    html_text = html._body.decode("utf-8")
    soup = BeautifulSoup(html_text, 'lxml')
    results = soup.select(".img-box")
    # list(soup.select(".img-box")[0].children)[1].attrs["href"]
    if len(results) > 0:
        first_url = list(results[0].children)[1].attrs["href"]
        first_url = "http://101.132.162.63" + first_url
    else:
        first_url = ""
    # print(first_url)
    return first_url


if __name__ == '__main__':
    get_one_click_url("汉堡包")
