import * as tf from '@tensorflow/tfjs';
const inputValorCalulcar = document.getElementById('valorACalcular');
const contenedorResultado = document.getElementById('resultado');
let modeloEntrenado;
let chart;
const calcularValoresY = (paramValoresX) => {
  const arrayResultadosY = [];
  for (let i = 0; i < paramValoresX.length; i++) {
    const y = paramValoresX[i] + 3;
    arrayResultadosY.push(y);
  }
  return arrayResultadosY;
};

const funcionLineal = async () => {
  contenedorResultado.innerHTML = 'El modelo se esta entrenando...';

  const model = tf.sequential();

  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

  const valoresInicialesX = [-1, 0, 1, 2, 3, 4];

  const resultadoY = calcularValoresY(valoresInicialesX);

  // console.log(resultadoY);

  const xs = tf.tensor2d(valoresInicialesX, [6, 1]);
  const ys = tf.tensor2d(resultadoY, [6, 1]);

  await model.fit(xs, ys, {
    epochs: 300,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        chart.series[0].addPoint(logs.loss);
      },
    },
  });

  inputValorCalulcar.disabled = false;
  inputValorCalulcar.focus();

  modeloEntrenado = model;
  contenedorResultado.innerHTML = 'Modelo entrenado, listo para usar';
};

document.addEventListener('DOMContentLoaded', () => {
  funcionLineal();

  chart = new Highcharts.chart('funcionPerdida', {
    title: {
      text: 'Datos de la función de perdida',
    },
    xAxis: {
      categories: 0,
    },
    series: [
      {
        name: 'Datos de la función de perdida',
        data: 0,
      },
    ],
    credits: {
      enabled: false,
    },
  });

  inputValorCalulcar.addEventListener('keyup', (event) => {
    if (event.keyCode === 13) {
      event.preventDefault();
      const valorACalcular = parseInt(inputValorCalulcar.value);
      const resultado = modeloEntrenado.predict(
        tf.tensor2d([valorACalcular], [1, 1])
      );

      const valorResultado = resultado.dataSync();
      // console.log(valorResultado);
      armarGrafica(valorACalcular, valorResultado[0]);
      contenedorResultado.innerHTML = `El resultado aproximado para Y es de: ${valorResultado}`;
    }
  });
});

const armarGrafica = (valorX, valorY) => {
  const trace1 = {
    x: [valorX],
    y: [valorY],
    mode: 'markers',
    // type: 'scatter',
  };

  // Define Data
  const data = [trace1];

  // Define Layout
  const layout = {
    xaxis: { range: [-40, 160], title: 'Valores de X' },
    yaxis: { range: [-20, 20], title: 'Valores de Y' },
    // title: 'House Prices vs. Size',
  };

  Plotly.newPlot('myPlot', data, layout);
};
