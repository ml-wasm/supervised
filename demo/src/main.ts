import init, {
  FloatsMatrix,
  FloatsVector,
  GaussianNaiveBayes,
} from '@ml.wasm/supervised';

const gaussianBayes = (
  xTrain: FloatsMatrix,
  xTest: FloatsMatrix,
  yTrain: FloatsVector,
  yTest: FloatsVector) => {
    const clf = new GaussianNaiveBayes();
    console.log(xTrain.nrows());
    const st = performance.now();
    clf.fit(xTrain, yTrain);
    const et = performance.now();
    console.log(`Training took ${et - st} ms`);
    
    const sp = performance.now();
    const preds = clf.predict(xTest);
    const ep = performance.now();
    console.log(`Prediction took took ${ep - sp} ms`);
    console.log(preds.data);

    const accuracy = clf.score(xTest, yTest);
    console.log(accuracy);
}

const fetchData = async (filenames: String[]): Promise<File[]> => {
  const array = [];
  for (const filename of filenames) {
    const text = await (await fetch(`../${filename}`)).text();
    array.push(new File([text], "a"));
  }
  return array;
}


(async () => {
  await init();

  const [ xTrainFile, xTestFile, yTrainFile, yTestFile ] = await fetchData(['x_tr.csv', 'x_te.csv', 'y_tr.csv', 'y_te.csv']);
  console.log("Files fetched");
  const xTrain = await FloatsMatrix.newFromCSV(xTrainFile);
  console.log("xTrain converted", xTrain instanceof FloatsMatrix);
  const xTest = await FloatsMatrix.newFromCSV(xTestFile);
  console.log("xTest converted", xTest instanceof FloatsMatrix);
  const yTrain = await FloatsVector.newFromCSV(yTrainFile);
  console.log("yTrain converted", yTrain instanceof FloatsVector);
  const yTest = await FloatsVector.newFromCSV(yTestFile);
  console.log("yTest converted", yTest instanceof FloatsVector);

  gaussianBayes(xTrain, xTest, yTrain, yTest);
})();


