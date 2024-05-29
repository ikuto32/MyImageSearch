import time
import urllib.error
import urllib.request
import tqdm

def main():
    for i in tqdm.tqdm(range(30)):
        try:
            with urllib.request.urlopen('https://source.unsplash.com/random') as web_file:
                data = web_file.read()
                with open(f"images/{i}.png", mode='wb') as local_file:
                    local_file.write(data)
        except urllib.error.URLError as e:
            print(e)
        time.sleep(1.0)

if __name__ == "__main__":
    main()