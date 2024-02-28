import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import math
# Ruta del archivo CSV
def cargar():
    archivo_csv = 'datos.csv'  

    datos = pd.read_csv(archivo_csv, delimiter=';', header=0, usecols=["decaimiento (us)"])

    for i in range(len(datos)):
        if datos.loc[i, "decaimiento (us)"]>=40000:
            datos = datos.drop(i)
    print(datos.head())

    
    return datos

def valores(datos):
    dic={}
    dic_ordenado={}
    for valor in datos["decaimiento (us)"]:
        if valor not in dic:
            dic[valor]=1
        else:
            dic[valor]+=1

    llaves_ordenadas = sorted(dic.keys())
    for llave in llaves_ordenadas:
        dic_ordenado[llave] = dic[llave]

    print(dic_ordenado)
    df=pd.DataFrame(list(dic_ordenado.items()), columns=['Llave', 'Valor'])
    nombre="TablaOrdenada.xlsx"
    df.to_excel(nombre, index=False)
    return dic_ordenado

def suma(dic_ordenado:dict):
    dic_suma={}
    suma=0
    llaves=dic_ordenado.keys()
    for llave in llaves:
        dic_suma[llave]=suma
        suma+=dic_ordenado[llave]
    print(dic_suma)
    return dic_suma

def resta(dic_suma:dict):
    dic_resta={}
    val=list(dic_suma.values())
    suma=val[-1]
    llaves = dic_suma.keys()
    for llave in llaves:
        dic_resta[llave]=suma
        suma=suma-dic_suma[llave]
    x=np.array(list(dic_resta.keys()))
    y=np.array(list(dic_resta.values()))
    y_lin=[]
    for valor in y:
        y_lin.append(math.log((valor/1491)+38000))
    
    def func(x, A, B):
        return (A*x)+B
    params, covariance = curve_fit(func, x, y_lin)

    A_fit, B_fit = params
    print("A =", A_fit)
    print("B =", B_fit)
    

    return dic_resta

def grafica(dic_resta:dict):
    x=np.array(list(dic_resta.keys()))
    y=np.array(list(dic_resta.values()))
    y_lin=[]
    y_cuadrados=[]
    error=np.array([0.001 for _ in range(len(x))])
    for valor in y:
        y_lin.append(math.log(valor/1491))
    def func(x, A, B):
        return (A*x)+B
    params, covariance = curve_fit(func, x, y_lin)

    A_fit, B_fit = params
    A_fit_error = np.sqrt(covariance[0, 0]) 
    B_fit_error = np.sqrt(covariance[1, 1])

    y_fit=func(x,A_fit, B_fit)
    def func_exp(x, C):
        return C*np.exp(-x/5326)
    param, cov = curve_fit(func_exp, x, y)
    C_fit = param
    C_fit_error = np.sqrt(cov[0, 0])

    y_exp=func_exp(x, C_fit)

    for i in range(len(x)):
        y_cuadrados.append(abs(y[i]-func_exp(x[i], C_fit)))

    print("A =", A_fit, "+/-", A_fit_error)
    print("B =", B_fit, "+/-", B_fit_error)
    print("C =", C_fit, "+/-", C_fit_error)
    plt.figure()
    plt.plot(x,y_fit)
    plt.plot(x,y_lin, marker='o', linestyle='')
    plt.title("Linealización datos experimentales")
    plt.xlabel("Tiempo (us)")
    plt.ylabel("Logaritmo natural del numero de particulas entre N0")



    fig, axs = plt.subplots(2, 1)
    axs[0].errorbar(x, y, xerr=error, fmt='o', label='Datos experimentales', alpha=0.5)
    axs[0].plot(x,y_exp, label="Linea de ajuste")
    axs[0].set_title("Decaimiento")
    axs[0].set_xlabel("Tiempo (us)")
    axs[0].set_ylabel("# de particulas")
    axs[0].legend()

    axs[1].scatter(x, y_cuadrados)
    axs[1].set_title('Error')
    axs[1].set_xlabel('Tiempo (us)')
    axs[1].set_ylabel('Residuos')

    plt.tight_layout()
    plt.show()

def ajuste():
    t_data = np.array([0, 1, 2, 3, 4, 5])
    y_data = np.array([5.1, 3.9, 2.8, 1.9, 1.3, 0.9])

    # Errores asociados a los datos (reemplaza esto con tus propios errores)
    errors = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    # Visualizar los resultados con barras de error
    plt.errorbar(t_data, y_data, yerr=errors, fmt='o', label='Datos experimentales')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def menu():
    cent=True
    datos=None
    while cent:
        print("Menu de opciones")
        print("")
        print(" 1 - Cargar datos")
        print(" 2 - valores")
        print(" 3 - Tabla de suma")
        print(" 4 - Tabla de resta")
        print(" 5 - Grafica")
        print(" 6 - Ajuste")
        print(" 7 - salir")


        opcion=int(input("Seleccione una opcion: "))
        if opcion == 1:
            datos=cargar()
            print("datos cargados con exito")
        elif opcion == 2:
            if datos is not None:
                dic_ordenado=valores(datos)
            else:
                print("Primero carga los datos")
        elif opcion == 3:
            dic_suma=suma(dic_ordenado)
        elif opcion == 4:
            dic_resta=resta(dic_suma)
        elif opcion == 5:
            grafica(dic_ordenado)
        elif opcion == 6:
            ajuste()
        elif opcion == 7:
            cent=False

menu()