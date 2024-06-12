from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import re
import json

application = Flask(__name__)
cors = CORS(application)
application.config["CORS_HEADERS"] = "content-type"

def r_raices(ecuacion):
    patron = r"sqrt\(([xy])\)"

    raices = re.findall(patron, ecuacion)

    for raiz in raices:
        ecuacion = ecuacion.replace(f"sqrt({raiz})", f"{raiz}^(1/2)")

    return ecuacion

def r_trigonometricas(ecuacion):
    patron = r"(sin|cos|tan)\(([xy])\)"

    funciones_trig = re.findall(patron, ecuacion)

    for funcion, variable in funciones_trig:
        if funcion == 'sin':
            ecuacion = ecuacion.replace(f"sin({variable})", f"{variable}")
        elif funcion == 'cos':
            ecuacion = ecuacion.replace(f"cos({variable})", f"{variable}")
        elif funcion == 'tan':
            ecuacion = ecuacion.replace(f"tan({variable})", f"{variable}")

    return ecuacion

def isHomogeneous(ecuacion):
    ecuacion_resuelta = r_raices(ecuacion)
    ecuacion_resuelta = r_trigonometricas(ecuacion_resuelta)

    patron = r"([+-]?\d*)(?:\*?([xy])(?:\^(-?\d+))?)?"

    matches = re.findall(patron, ecuacion_resuelta)

    exponentes_x = 0
    exponentes_y = 0

    for match in matches:
        coeficiente, variable, exponente = match
        if exponente:
            exponente = eval(exponente)
        else:
            exponente = 1
        if variable == 'x':
            exponentes_x += exponente
        elif variable == 'y':
            exponentes_y += exponente

    if exponentes_x == exponentes_y:
        return True
    else:
        return False

def euler(f, x0, y0, N, xf):
    h = abs(xf - x0) / N

    x = np.linspace(x0, xf, N+1)

    y = np.zeros(N + 1)

    y[0] = y0

    for i in range(1, N + 1):
        if x[i - 1] != 0:
            y[i] = y[i - 1] + h * f(x[i - 1], y[i - 1])
        else:
            y[i] = y[i - 1]
    return x, y

def runge_kutta(f, x0, y0, N, xf):
    h = abs(xf - x0) / N

    x = np.linspace(x0, xf, N+1)

    y = np.zeros(N + 1)

    y[0] = y0

    for i in range(1, N + 1):
        if x[i - 1] != 0:
            k1 = h * f(x[i - 1], y[i - 1])
            k2 = h * f(x[i - 1] + h / 2, y[i - 1] + k1 / 2)
            k3 = h * f(x[i - 1] + h / 2, y[i - 1] + k2 / 2)
            k4 = h * f(x[i - 1] + h, y[i - 1] + k3)
            y[i] = y[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        else:
            y[i] = y[i - 1]

    return x, y

def taylor_method(f, x0, y0, N, xf):
    h = abs(xf - x0) / (2 * N)

    x = np.linspace(x0, xf, N+1)

    y = np.zeros(N + 1)

    y[0] = y0

    for i in range(1, N + 1):
        if x[i - 1] != 0:
            y[i] = y[i - 1] + h * f(x[i - 1], y[i - 1])
        else:
            y[i] = y[i - 1]

    return x, y

def generate_method_data(x, y):
    result_x = []
    result_y = []
    errors = []
    for i in range(len(x)):
        result_x.append(round(x[i], 4))
        if np.isfinite(y[i]):
            result_y.append(round(y[i], 4))
            errors.append(None)
        else:
            result_y.append(None)
            errors.append("Valor no válido (división por cero)")
    return {"x": result_x, "y": result_y, "errors": errors}

def generate_solution_json(taylor_data, euler_data, runge_kutta_data):
    result = {
        "result": {
            "metodo_taylor": generate_method_data(taylor_data[0], taylor_data[1]),
            "metodo_euler": generate_method_data(euler_data[0], euler_data[1]),
            "metodo_runge": generate_method_data(runge_kutta_data[0], runge_kutta_data[1])
        }
    }
    return json.dumps(result, ensure_ascii=False, indent=4)

@application.route('/analize_equation', methods=['POST'])
def analizar_ecuacion():
    data = request.get_json()
    equation = data['expr']

    homogeneous = isHomogeneous(equation)

    if not homogeneous:
        result = {
        'ecuacion': equation,
        'message': 'La ecuación no es homogenea'
        }
        return jsonify(result)

    eq = eval('lambda x, y: ' + equation.replace('^', '**'))

    # Definir condiciones iniciales y parámetros
    x0 = 0  # Punto inicial
    y0 = 1  # Valor inicial de y en x0
    N = 10  # Número de pasos
    xf = 1  # Punto final

    x_euler, y_euler = euler(eq, x0, y0, N, xf)

    x_rk, y_rk = runge_kutta(eq, x0, y0, N, xf)

    x_taylor, y_taylor = taylor_method(eq, x0, y0, N, xf)

    return generate_solution_json((x_taylor, y_taylor), (x_euler, y_euler), (x_rk, y_rk))

if __name__ == '__main__':
    application.run(debug=True)
