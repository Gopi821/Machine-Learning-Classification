import mysql.connector

conn = mysql.connector.connect(host='localhost', user='root', password='12345')

mycursor = conn.cursor()

mycursor.execute("show databases")
for x in mycursor:
    print(x)
