bir <- readline("Bir sayı giriniz: ")

kalan <- as.numeric(bir)%%2

x <- ifelse(kalan == 0, "Çift", "Tek")
print(x)