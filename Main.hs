import Mnist
import Runner
import Backprop

import Numeric.LinearAlgebra
import Data.List
import System.Random

learningRate = 0.007

smallRandoms :: Int -> [Double]
smallRandoms seed = map (/100) (randoms (mkStdGen seed))

randomWeightMatrix :: Int -> Int -> Int -> Matrix Double
randomWeightMatrix numInputs numOutputs seed = (numOutputs><numInputs) weights
    where weights = take (numOutputs*numInputs) (smallRandoms seed)

zeroWeightMatrix :: Int -> Int -> Matrix Double
zeroWeightMatrix numInputs numOutputs = (numOutputs><numInputs) weights
    where weights = repeat 0


main :: IO ()
main = do
  let w1 = randomWeightMatrix (28*28 + 1) 20 7
  let w2 = randomWeightMatrix 20 10 42
  let initialNet = buildBackpropNet learningRate [w1, w2] tanhAS
  trainingData2 <- readTrainingData
  let trainingData = take 20000 trainingData2
  putStrLn $ "Training with " ++ show (length trainingData) ++ " images"
  let finalNet = trainWithAllPatterns initialNet trainingData
  testData2 <- readTestData
  let testData = take 1000 testData2
  putStrLn $ "Testing with " ++ show (length testData) ++ " images"
  let results = evalAllPatterns finalNet testData
  let score = fromIntegral (sum results)
  let count = fromIntegral (length testData)
  let percentage = 100.0 * score / count
  putStrLn $ "I got " ++ show percentage ++ "% correct"
