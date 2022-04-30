import { MnistData } from "./data.js";

async function showExamples(data) {
  // Create a container in the visor
  const surface = tfvis
    .visor()
    .surface({ name: "Input Data Examples", tab: "Input Data" });

  // Get the examples
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];

  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });

    const canvas = document.createElement("canvas");
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = "margin: 4px;";
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}

function getModel() {
  const model = tf.sequential();

  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const IMAGE_CHANNELS = 1;

  // In the first layer of our convolutional neural network we have
  // to specify the input shape. Then we specify some parameters for
  // the convolution operation that takes place in this layer.
  model.add(
    tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: "relu",
      kernelInitializer: "varianceScaling",
    })
  );

  // The MaxPooling layer acts as a sort of downsampling using max values
  // in a region instead of averaging.
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

  // Repeat another conv2d + maxPooling stack.
  // Note that we have more filters in the convolution.
  model.add(
    tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: "relu",
      kernelInitializer: "varianceScaling",
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  // it for input into our last layer. This is common practice when feeding
  // higher dimensional data to a final classification output layer.
  model.add(tf.layers.flatten());

  // Our last layer is a dense layer which has 10 output units, one for each
  // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
  const NUM_OUTPUT_CLASSES = 10;
  model.add(
    tf.layers.dense({
      units: NUM_OUTPUT_CLASSES,
      kernelInitializer: "varianceScaling",
      activation: "softmax",
    })
  );

  // Choose an optimizer, loss function and accuracy metric,
  // then compile and return the model
  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

async function train(model, data) {
  const metrics = ["loss", "val_loss", "acc", "val_acc"];
  const container = {
    name: "Model Training",
    tab: "Model",
    styles: { height: "1000px" },
  };
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

  const BATCH_SIZE = 512;
  const TRAIN_DATA_SIZE = 5500;
  const TEST_DATA_SIZE = 1000;

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 10,
    shuffle: true,
    callbacks: fitCallbacks,
  });
}

const classNames = [
  "Zero",
  "One",
  "Two",
  "Three",
  "Four",
  "Five",
  "Six",
  "Seven",
  "Eight",
  "Nine",
];

function doPrediction(model, data, testDataSize = 1) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const testData = data.nextTestBatch(testDataSize);
  console.log(testData);
  const testxs = testData.xs.reshape([
    testDataSize,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    1,
  ]);
  window.tfModel = model;
  console.log("testxs with 1 example", testxs);
  const labels = testData.labels.argMax(-1);
  const preds = model.predict(testxs).argMax(-1);

  testxs.dispose();
  return [preds, labels];
}

async function showAccuracy(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = { name: "Accuracy", tab: "Evaluation" };
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

  labels.dispose();
}

async function showConfusion(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = { name: "Confusion Matrix", tab: "Evaluation" };
  tfvis.render.confusionMatrix(container, {
    values: confusionMatrix,
    tickLabels: classNames,
  });

  labels.dispose();
}
let model;
async function run() {
  const data = new MnistData();
  await data.load();
  await showExamples(data);

  model = getModel();
  tfvis.show.modelSummary({ name: "Model Architecture", tab: "Model" }, model);

  await train(model, data);
  await showAccuracy(model, data);
  await showConfusion(model, data);
}

const runButton = document.getElementById("run-button");
const categorizeButton = document.getElementById("categorize-button");
async function runButtonClick() {
  runButton.disabled = true;
  await run();
  runButton.disabled = false;
  categorizeButton.disabled = false;
}
runButton.addEventListener("click", runButtonClick);

const c = document.getElementById("canvas");
c.addEventListener("mousedown", setLastCoords);
c.addEventListener("mousemove", freeForm);

const ctx = c.getContext("2d");
let lastX = 0;
let lastY = 0;

function setLastCoords(e) {
  const { x, y } = c.getBoundingClientRect();
  lastX = (e.clientX - x) / 5;
  lastY = (e.clientY - y) / 5;
}
function freeForm(e) {
  if (e.buttons !== 1) return; // left button is not pushed yet
  penTool(e);
}
function penTool(e) {
  const { x, y } = c.getBoundingClientRect();
  const newX = (e.clientX - x) / 5;
  const newY = (e.clientY - y) / 5;

  ctx.beginPath();
  ctx.lineWidth = 3;
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(newX, newY);
  ctx.strokeStyle = "white";
  ctx.stroke();
  ctx.closePath();

  lastX = newX;
  lastY = newY;
}
function clearCanvas() {
  ctx.clearRect(0, 0, c.width, c.height);
}
document.getElementById("clear-button").addEventListener("click", clearCanvas);
function getDataArr(canvas) {
  const ctx = canvas.getContext("2d");
  const dataArr = ctx
    .getImageData(0, 0, 28, 28)
    .data.filter((el, i) => i % 4 === 3);
  return dataArr;
}
function getDataMatrix(canvas) {
  const dataArr = getDataArr(canvas);
  // group into rows of length 28
  const dataMatrix = [];
  for (let i = 0; i < dataArr.length; i += 28) {
    dataMatrix.push(dataArr.slice(i, i + 28));
  }
  return dataMatrix;
}

function doPredictFromCanvas(model, canvas) {
  const dataArr = getDataArr(canvas);
  const t = tf.tensor2d(Uint8Array.from(dataArr), [1, 784], "int32");
  const testxs = t.reshape([1, 28, 28, 1]);
  const preds = model.predict(testxs); //.argMax(-1);
  testxs.dispose();
  return preds;
}
categorizeButton.addEventListener("click", async () => {
  const m = getDataMatrix(c);
  const drawingDataContainer = document.getElementById(
    "drawing-data-container"
  );
  drawingDataContainer.innerHTML = "";
  m.forEach((row, i) => {
    const rowContainer = document.createElement("div");
    rowContainer.style.height = "10px";
    row.forEach((el, j) => {
      const elContainer = document.createElement("div");
      elContainer.style.display = "inline-block";
      elContainer.style.width = "10px";
      elContainer.style.height = "5px";
      elContainer.style.fontSize = "6px";
      elContainer.innerText = el.toString();
      rowContainer.appendChild(elContainer);
    });
    drawingDataContainer.appendChild(rowContainer);
  });
  const preds = doPredictFromCanvas(model, c);
  const categorizeResultElem = document.getElementById("categorize-result");
  categorizeResultElem.innerHTML =
    preds.toString() + "<br>" + preds.argMax(-1).toString();
});
