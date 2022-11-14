import os
import pprint
import time
import urllib.error
import urllib.request

def main():
    for i in range(100):
        try:
            with urllib.request.urlopen('https://source.unsplash.com/random') as web_file:
                data = web_file.read()
                print()
                with open(f"images/{i}.png", mode='wb') as local_file:
                    local_file.write(data)
        except urllib.error.URLError as e:
            print(e)
        time.sleep(0.5)

if __name__ == "__main__":
    main()