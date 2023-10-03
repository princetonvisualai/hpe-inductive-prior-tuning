import time

i = 0

while True:

    f = open("demo.txt", "w")
    f.write(str(i))
    f.close()
    i += 1
    time.sleep(10)