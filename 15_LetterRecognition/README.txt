Установить пакет mlbench и решить одноименную задачу. 

Описание задачи и переменных можно прочитать в справке по mlbench и в файле mlbench.pdf


Установка пакета:

install.packages("mlbench") 

Удобнее использовать IDE RStudio, меню Tools.

Загрузка пакета в оперативную память:

library(mlbench)

Загрузка память набора данных, например "BreastCancer":

data("BreastCancer")

Справка по набору данных:

help("BreastCancer")
