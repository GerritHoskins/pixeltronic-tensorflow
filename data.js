const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node'); // Use '@tensorflow/tfjs-node-gpu' for GPU acceleration

// Assuming you have a dataset object like this:
// dataset = [{imagePath: 'path/to/image1.jpg', label: 'happy'}, {...}]

// Function to read an image from file path and convert it to a tensor
function readImage(path) {
    const imageBuffer = fs.readFileSync(path);
    const tfImage = tf.node.decodeImage(imageBuffer);
    return tfImage;
}

// Function to convert a label to one-hot encoding
function oneHotEncode(label, classes) {
    return tf.oneHot(classes.indexOf(label), classes.length);
}

// Main function to load data
async function loadData(dataset, classes) {
    const images = [];
    const labels = [];

    for (let i = 0; i < dataset.length; i++) {
        const imagePath = dataset[i].imagePath;
        const label = dataset[i].label;

        const image = readImage(imagePath);
        images.push(image);

        const encodedLabel = oneHotEncode(label, classes);
        labels.push(encodedLabel);
    }

    // Convert arrays to tensors and resize images
    const imagesTensor = tf.stack(images).resizeBilinear([224, 224]); // Resize to the expected input size for MobileNet
    const labelsTensor = tf.stack(labels);

    return {
        images: imagesTensor,
        labels: labelsTensor
    };
}

// Example usage
const classes = ['happy', 'scared', 'aggressive']; // List of all classes
loadData(myDataset, classes).then(processedData => {
    console.log('Data loaded and processed');
});
