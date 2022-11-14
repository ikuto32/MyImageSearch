


def printTest(text :str):
    print(text)


#標準入力を取得し、IDごとに振り分けるクラス
class MessageReceiver:

    def __init__(self):

        #IDをキーとした、辞書。パラメータを取得中のイベント
        self.listeningEvents = {}

        pass

    #標準入力を待機。引数はコールバック関数(id, params) -> None
    #標準入力によってパラメータを全て与えられたときに呼び出される
    def listenStdin(self, callbackOnInputEnd) -> None:

        while True:

            #標準入力をパースし、パラメータを取得
            line = input()
            array = line.split(",")

            #パラメータの情報を取得
            id :str = array[0]
            key :str = array[1]
            value :str = ",".join(array[2:])
            
            print("受信: " + id + ", " + key + ", " + value)

            #パラメータの入力が始まったとき、IDに対応するイベントオブジェクトを生成
            if key == "param-start" and id not in self.listeningEvents: 

                self.listeningEvents[id] = Event(id)
                continue

            #パラメータの入力が突然開始されたとき、エラーを報告
            if id not in self.listeningEvents:

                print(id + ",java-error:[param-start]が実行されていません")
                break
            
            #イベントオブジェクトを取得
            event :Event = self.listeningEvents[id]

            #パラメータの入力が終わったとき、コールバック関数を呼び出す
            if key == "param-end":

                #コールバック関数を呼び出し、イベントオブジェクトを削除する
                callbackOnInputEnd(event)
                del self.listeningEvents[id]
                continue
            
            #パラメータを追加
            setattr(event.params, key, value)



#イベントのパラメータを示すクラス
class Params:
    def __str__(self):
        return str(self.__dict__)

#イベントの情報を表すクラス
class Event:

    def __init__(self, id :str):
        self.id = id
        self.params = Params()

    #値を送信する
    def print(self, text):
    
        print(self.id + "," + str(text))

    #値を送信する
    def sendJavaOutput(self, text):

        print(self.id + ",java-output:" + str(text))

    #実行してもらう動作を送信する
    def sendJavaAction(self, text):

        print(self.id + ",java-action:" + str(text))


