import requests
import os
import re
import random
import time
import csv

# request header
HEADERS = {'Host': 'api.taptapdada.com',
           'Connection': 'Keep-Alive',
           'Accept-Encoding': 'gzip',
           'User-Agent': 'okhttp/3.10.0'}
# basic webpage contents 10 pages
BASE_URL = 'https://api.taptapdada.com/review/v1/by-app?sort=new&app_id={}' \
            '&X-UA=V%3D1%26PN%3DTapTap%26VN_CODE%3D593%26LOC%3DCN%26LANG%3Dzh_CN%26CH%3Ddefault' \
            '%26UID%3D8a5b2b39-ad33-40f3-8634-eef5dcba01e4%26VID%3D7595643&from={}'
# save stop_point text
STOP_POINT_FILE = 'stop_point.txt'


class TapSpiderByRequests:
    def __init__(self, csv_save_path, game_id):
        # get stop_point
        self.start_from = self.resume()
        # reset save list
        self.reviews = []
        # run spider
        self.spider(csv_save_path, game_id)

    def spider(self, csv_save_path, game_id):
        end_from = self.start_from + 300
        # 30 pages a batch
        for i in range(self.start_from, end_from+1, 10):
            url = BASE_URL.format(game_id, i)
            try:
                resp = requests.get(url, headers=HEADERS).json()
                resp = resp.get('data').get('list')
                self.parse_info(resp)
                print('=============on %d pages=============' % int(i/10))

                # wait for 0-2 second for next step
                if i != end_from:
                    pause = random.uniform(0, 2)
                    time.sleep(pause)
                # save the crawl data
                else:
                    with open(STOP_POINT_FILE, 'w') as f:
                        f.write(str(i+10))

            # If error occurs save exist data and strat crawl again
            except Exception as error:
                with open(STOP_POINT_FILE, 'w') as f:
                    f.write(str(i))
                # print exception error
                print('%i pages error occurs please try again' % int(i/10))
                raise error
                # exit
                exit()

        # save it as csv
        self.write_csv(csv_save_path, self.reviews)

    def parse_info(self, resp):
        for r in resp:
            review = {}
            # id
            review['id'] = r.get('id')
            # name
            review['author'] = r.get('author').get('name').encode('gbk', 'ignore').decode('gbk')
            # update_time
            timeArray=time.localtime(r.get('updated_time'))
            otherStyleTime=time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
            review['updated_time'] = otherStyleTime
            # play_time
            review['spent'] = r.get('spent')
            # score
            review['stars'] = r.get('score')
            # comment
            content = r.get('contents').get('text').strip()
            review['contents'] = re.sub('<br />|&nbsp', '', content).encode('gbk', 'ignore').decode('gbk')

            self.reviews.append(review)

    # stop point resume
    def resume(self):
        start_from = 0
        if os.path.exists(STOP_POINT_FILE):
            with open(STOP_POINT_FILE, 'r') as f:
                start_from = int(f.readline())
        return start_from

    # write into csv
    def write_csv(self, full_path, reviews):
        title = reviews[0].keys()
        path, file_name = os.path.split(full_path)
        if os.path.exists(full_path):
            with open(full_path, 'a+', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, title)
                writer.writerows(reviews)
        else:
            try:
                os.mkdir(path)
            except Exception:
                print('Roots already exist')
            with open(full_path, 'a+', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, title)
                writer.writeheader()
                writer.writerows(reviews)


if __name__ == '__main__':
    # csv save root
    PATH = r"C:\\Users\\Tony's PC\\Desktop\\NLP\\Project Of NLP\\Python_NLP\\data\\TapTap\\TapTap_data.csv"
    
    game_id = 168332
    # get 15000 comments
    for i in range(50):
        TapSpiderByRequests(PATH, game_id)



'''This programme basically from https://github.com/sariel-black/taptap_emotion_analyse/blob/master/taptap%E8%AF%84%E8%AE%BA%E6%83%85%E6%84%9F%E5%88%86%E6%9E%90/taptap%E8%AF%84%E8%AE%BA%E7%88%AC%E5%8F%96/tap%20spider.py
'''
