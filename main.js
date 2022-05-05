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
// const margin = { top: 10, right: 50, bottom: 50, left: 50 },
//   width = 450 - margin.left - margin.right,
//   height = 400 - margin.top - margin.bottom;

// const svg = d3
//   .select('#root')
//   .attr('width', width + margin.left + margin.right)
//   .attr('height', height + margin.top + margin.bottom)
//   .append('g')
//   .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

// // Define chart area
// svg
//   .append('clipPath')
//   .attr('id', 'chart-area')
//   .append('rect')
//   .attr('x', 0)
//   .attr('y', 0)
//   .attr('width', width)
//   .attr('height', height);

// // Add Axes
// const xMax = 4;
// const yMax = 5;

// let xScale = d3.scaleLinear([0, xMax], [0, width]);
// let yScale = d3.scaleLinear([0, yMax], [height, 0]);

// let xAxis = d3.axisBottom(xScale);
// let yAxis = d3.axisLeft(yScale);
// svg.append('g').attr('transform', `translate(0,${height})`).call(xAxis);
// svg.append('g').attr('transform', `translate(0,0)`).call(yAxis);

// // Axes label
// svg
//   .append('text')
//   .attr('class', 'x label')
//   .attr('text-anchor', 'end')
//   .attr('x', width / 2 + 5)
//   .attr('y', height + 35)
//   .text('x');

// svg
//   .append('text')
//   .attr('class', 'y label')
//   .attr('text-anchor', 'end')
//   .attr('y', -35)
//   .attr('x', -height / 2)
//   .attr('transform', 'rotate(-90)')
//   .html('y');

// function f(x) {
//   return x + 3;
// }

// function graphFunction() {
//   let pointNum = 500;

//   const data = [];
//   for (let x = 0; x <= pointNum; x++) {
//     let y = f(x);
//     data.push([x, y]);
//   }
//   return data;
// }

// // Add function graph
// let line = d3
//   .line()
//   .x((d) => xScale(d[0]))
//   .y((d) => yScale(d[1]));
// svg
//   .append('path')
//   .datum(graphFunction())
//   .attr('clip-path', 'url(#chart-area)')
//   .attr('fill', 'none')
//   .attr('stroke', 'teal')
//   .attr('stroke-width', 2)
//   .attr('d', line);
