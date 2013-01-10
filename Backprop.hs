module Backprop
  (BackpropNet, buildBackpropNet, logisticSigmoidAS, tanhAS, identityAS)
    where

import MatrixPlus as P
import NeuralNet

import Numeric.LinearAlgebra as N
import Test.QuickCheck
import Control.Exception
import Debug.Trace

-- {{{ Layers

-- | An individual layer in a neural network, prior to propagation
data Layer = Layer
    {
      -- The weights for this layer
      lW :: Matrix Double,
      -- The activation specification for this layer
      lAS :: ActivationSpec
    }

instance Show Layer where
    show layer = "w=" ++ show (lW layer) ++ ", activation spec=" ++ show (lAS layer)

-- | Generate a layer of the specified "size", with arbitrary values.
-- | To see sample values (of "size" 4), in GHCi type: sample (sizedArbLayer 4)
sizedArbLayer :: Int -> Gen Layer
sizedArbLayer n = do
    -- QuickCheck passes a value n >= 0, but we can't create a vector of length 0, so use n+1
    r <- choose(1, n+1)
    c <- choose(1, n+1)
    w <- sizedArbWeightMatrix r c
    s <- (arbitrary :: Gen ActivationSpec)
    return Layer{ lW=w, lAS=s }

-- | Generate a layer of the specified "size" and the specified input width, with arbitrary values.
-- | To see sample values (of input width 3, "size" 4), in GHCi type: sample (sizedArbLayer' 3 4)
sizedArbLayer' :: Int -> Int -> Gen Layer
sizedArbLayer' c n = do
    -- QuickCheck passes a value n >= 0, but we can't create a vector of length 0, so use n+1
    r <- choose(1, n+1)
    w <- sizedArbWeightMatrix r c
    s <- (arbitrary :: Gen ActivationSpec)
    return Layer{ lW=w, lAS=s }

-- | Generate a layer of arbitrary "size", with arbitrary values.
-- | To see some sample values, in GHCi type: sample' arbitrary :: IO [Layer]
instance Arbitrary Layer where
    arbitrary = sized sizedArbLayer

inputWidth :: Layer -> Int
inputWidth = cols . lW

outputWidth :: Layer -> Int
outputWidth = rows . lW

-- }}}

-- {{{ Propagation

-- | An individual layer in a neural network, after propagation but prior to backpropagation
data PropagatedLayer
    = PropagatedLayer
        {
          -- The input to this layer
          pIn :: ColumnVector Double,
          -- The output from this layer
          pOut :: ColumnVector Double,
          -- The value of the first derivative of the activation function for this layer
          pF'a :: ColumnVector Double,
          -- The weights for this layer
          pW :: Matrix Double,
          -- The activation specification for this layer
          pAS :: ActivationSpec
        }
    | PropagatedSensorLayer
        {
          -- The output from this layer
          pOut :: ColumnVector Double
        }

instance Show PropagatedLayer where
    show (PropagatedLayer x y f'a w s) = 
        "in=" ++ show x
        ++ ", out=" ++ show y
        ++ ", f'(a)=" ++ show f'a
        ++ ", w=" ++ show w 
        ++ ", " ++ show s
    show (PropagatedSensorLayer x) = "out=" ++ show x

    -- {{{ Testing

{-
-- | Generate a propagated hidden or output layer of the specified "size", with arbitrary values.
-- | To see sample values (of "size" 4), in GHCi type: sample (sizedArbPropagatedLayer 4)
sizedArbPropagatedLayer :: Int -> Gen PropagatedLayer
sizedArbPropagatedLayer n = do
    -- QuickCheck passes a value n >= 0, but we can't create a vector of length 0, so use n+1
    r <- choose(1, n+1)
    c <- choose(1, n+1)
    w <- sizedArbWeightMatrix r c
    y <- sizedArbColumnVector r
    return PropagatedLayer{ pOut=y
                          , pF'a=(zeroColumnVector r)
                          , pW=w
                          , pF=id
                          , pF'=id'
                          }

-- | Generate a propagated sensor layer of the specified "size", with arbitrary values.
-- | To see sample values (of "size" 4), in GHCi type: sample (sizedArbPropagatedLayer2 4)
sizedArbPropagatedLayer0 :: Int -> Gen PropagatedLayer
sizedArbPropagatedLayer0 n = do
    -- QuickCheck passes a value n >= 0, but we can't create a vector of length 0, so use n+1
    r <- choose(1, n+1)
    y <- sizedArbColumnVector r
    return PropagatedSensorLayer{ pOut=y }


-- | Generate a propagated layer of arbitrary "size", with arbitrary values.
-- | To see some sample values, in GHCi type: sample' arbitrary :: IO [PropagatedLayer]
instance Arbitrary PropagatedLayer where
    arbitrary = frequency
        [ (4, sized sizedArbPropagatedLayer)
        , (1, sized sizedArbPropagatedLayer0) ]
-}
    -- }}}

-- | Propagate the inputs through this layer to produce an output.
propagate :: PropagatedLayer -> Layer -> PropagatedLayer
propagate layerJ layerK = PropagatedLayer
        {
          pIn = x,
          pOut = y,
          pF'a = f'a,
          pW = w,
          pAS = lAS layerK
        }
  where x = pOut layerJ
        w = lW layerK
        a = w <> x
        f = asF ( lAS layerK )
        y = P.mapMatrix f a
        f' = asF' ( lAS layerK )
        f'a = P.mapMatrix f' a


-- }}}

-- {{{ Backpropagation

-- | An individual layer in a neural network, after backpropagation
data BackpropagatedLayer = BackpropagatedLayer
    {
      -- Del-sub-z-sub-l of E
      bpDazzle :: ColumnVector Double,
      -- The error due to this layer
      bpErrGrad :: ColumnVector Double,
      -- The value of the first derivative of the activation 
      --   function for this layer
      bpF'a :: ColumnVector Double,
      -- The input to this layer
      bpIn :: ColumnVector Double,
      -- The output from this layer
      bpOut :: ColumnVector Double,
      -- The weights for this layer
      bpW :: Matrix Double,
      -- The activation specification for this layer
      bpAS :: ActivationSpec
    }

instance Show BackpropagatedLayer where
    show layer =
        "dazzle=" ++ show (bpDazzle layer)
        ++ ", grad=" ++ show (bpErrGrad layer)
        ++ ", in=" ++ show (bpIn layer)
        ++ ", out=" ++ show (bpOut layer)
        ++ ", w=" ++ show (bpW layer) 
        ++ ", activationFunction=?, activationFunction'=?"

backpropagateFinalLayer ::
    PropagatedLayer -> ColumnVector Double -> BackpropagatedLayer
backpropagateFinalLayer l t = BackpropagatedLayer
    {
      bpDazzle = dazzle,
      bpErrGrad = errorGrad dazzle f'a (pIn l),
      bpF'a = pF'a l,
      bpIn = pIn l,
      bpOut = pOut l,
      bpW = pW l,
      bpAS = pAS l
    }
    where dazzle =  pOut l - t
          f'a = pF'a l

errorGrad :: ColumnVector Double -> ColumnVector Double -> ColumnVector Double
    -> ColumnVector Double
errorGrad dazzle f'a input = (dazzle * f'a) <> trans input

-- | Propagate the inputs backward through this layer to produce an output.
backpropagate :: PropagatedLayer -> BackpropagatedLayer -> BackpropagatedLayer
backpropagate layerJ layerK = BackpropagatedLayer
    {
      bpDazzle = dazzleJ,
      bpErrGrad = errorGrad dazzleJ f'aJ (pIn layerJ),
      bpF'a = pF'a layerJ,
      bpIn = pIn layerJ,
      bpOut = pOut layerJ,
      bpW = pW layerJ,
      bpAS = pAS layerJ
    }
    where dazzleJ = wKT <> (dazzleK * f'aK)
          dazzleK = bpDazzle layerK
          wKT = trans ( bpW layerK )
          f'aK = bpF'a layerK
          f'aJ = pF'a layerJ


-- }}}

-- {{{ Adjusting weights after backpropagation

update :: Double -> BackpropagatedLayer -> Layer
update rate layer = Layer
        {
          lW = wNew,
          lAS = bpAS layer
        }
    where wOld = bpW layer
          delW = rate `scale` bpErrGrad layer
          wNew = wOld - delW

-- }}}

-- {{{ Building a network

data BackpropNet = BackpropNet
    {
      layers :: [Layer],
      learningRate :: Double
    } deriving Show

buildBackpropNet ::
  -- The learning rate
  Double ->
  -- The weights for each layer
  [Matrix Double] ->
  -- The activation specification (used for all layers)
  ActivationSpec ->
  -- The network
  BackpropNet
buildBackpropNet lr ws s = BackpropNet { layers=ls, learningRate=lr }
  where checkedWeights = scanl1 checkDimensions ws
        ls = map buildLayer checkedWeights
        buildLayer w = Layer { lW=w, lAS=s }

checkDimensions :: Matrix Double -> Matrix Double -> Matrix Double
checkDimensions w1 w2 =
  if rows w1 == cols w2
       then w2
       else error "Inconsistent dimensions in weight matrix"

propagateNet :: ColumnVector Double -> BackpropNet -> [PropagatedLayer]
propagateNet input net = tail calcs
  where calcs = scanl propagate layer0 (layers net)
        layer0 = PropagatedSensorLayer{ pOut=validatedInputs }
        validatedInputs = validateInput net input

validateInput :: BackpropNet -> ColumnVector Double -> ColumnVector Double
validateInput net = validateInputValues . validateInputDimensions net

validateInputDimensions ::
    BackpropNet ->
    ColumnVector Double ->
    ColumnVector Double
validateInputDimensions net input =
  if got == expected
       then input
       else error ("Input pattern has " ++ show got ++ " bits, but " ++ show expected ++ " were expected")
           where got = rows input
                 expected = inputWidth (head (layers net))

validateInputValues :: ColumnVector Double -> ColumnVector Double
validateInputValues input =
  if (min >= 0) && (max <= 1)
       then input
       else error "Input bits outside of range [0,1]"
       where min = minimum ns
             max = maximum ns
             ns = toList ( flatten input )

backpropagateNet :: 
  ColumnVector Double -> [PropagatedLayer] -> [BackpropagatedLayer]
backpropagateNet target layers = scanr backpropagate layerL hiddenLayers
  where hiddenLayers = init layers
        layerL = backpropagateFinalLayer (last layers) target

-- }}}

-- {{{ Define BackpropNet to be an instance of Neural Net

instance NeuralNet BackpropNet where
  evaluate = evaluateBPN
  train = trainBPN

evaluateBPN :: BackpropNet -> [Double] -> [Double]
evaluateBPN net input = columnVectorToList( pOut ( last calcs ))
  where calcs = propagateNet x net
        x = listToColumnVector (1:input)

trainBPN :: BackpropNet -> [Double] -> [Double] -> BackpropNet
trainBPN net input target = BackpropNet { layers=newLayers, learningRate=rate }
  where newLayers = map (update rate) backpropagatedLayers
        rate = learningRate net
        backpropagatedLayers = backpropagateNet (listToColumnVector target) propagatedLayers
        propagatedLayers = propagateNet x net
        x = listToColumnVector (1:input)

-- }}}

-- {{{ General Testing

-- | A layer with suitable input and target vectors, suitable for testing.
data LayerTestData = LTD (ColumnVector Double) Layer (ColumnVector Double)
  deriving Show

-- | Generate a layer with suitable input and target vectors, of the specified "size", 
-- | with arbitrary values.
-- | To see sample values (of "size" 4), in GHCi type: sample (sizedLayerTestData 4)
sizedLayerTestData :: Int -> Gen LayerTestData
sizedLayerTestData n = do
    l <- sizedArbLayer n
    x <- sizedArbColumnVector (inputWidth l)
    t <- sizedArbColumnVector (outputWidth l)
    return (LTD x l t)

instance Arbitrary LayerTestData where
  -- | To see sample values, in GHCi type: sample arbLayerTestData
  arbitrary = sized sizedLayerTestData

-- | Training reduces error in the final (output) layer
prop_trainingReducesFinalLayerError :: LayerTestData -> Property
prop_trainingReducesFinalLayerError (LTD x l t) =
    -- (collect l) . -- uncomment to view test data
    (classifyRange "len x " n 0 25) .
    (classifyRange "len x " n 26 50) . 
    (classifyRange "len x " n 51 75) . 
    (classifyRange "len x " n 76 100) $
    errorAfter < errorBefore || errorAfter < 0.01
        where n = inputWidth l
              pl0 = PropagatedSensorLayer{ pOut=x }
              pl = propagate pl0 l
              bpl = backpropagateFinalLayer pl t
              errorBefore = P.magnitude (t - pOut pl)
              lNew = update 0.0000000001 bpl -- make sure we don't overshoot the mark
              plNew = propagate pl0 lNew
              errorAfter =  P.magnitude (t - pOut plNew)

iterateTraining n pl0 r ltd = pOut plFinal
    where iterations = iterate (trainOneLayer pl0 r) ltd
          (LTD _ lFinal _) = last (take n iterations)
          plFinal = propagate pl0 lFinal

trainOneLayer :: PropagatedLayer -> Double -> LayerTestData -> LayerTestData 
trainOneLayer pl0 r (LTD x l t) = LTD x lNew t
    where pl = propagate pl0 l
          bpl = backpropagateFinalLayer pl t
          lNew = update r bpl

-- | Testable property:
-- | Training a single layer with the same input repeatedly will eventually yield the target
prop_trainingOneLayerWithOneInputYieldsPerfection :: LayerTestData -> Property
prop_trainingOneLayerWithOneInputYieldsPerfection (LTD x l t) =
    -- (collect l) . -- uncomment to view test data
    (classifyRange "len x " n 0 25) . 
    (classifyRange "len x " n 26 50) . 
    (classifyRange "len x " n 51 75) .
    (classifyRange "len x " n 76 100) $
    e < 0.1
        where n = inputWidth l
              r = 1
              pl0 = PropagatedSensorLayer{ pOut=x }
              y = iterateTraining 100 pl0 r (LTD x l t)
              e = P.magnitude (t - y)

-- | A layer with suitable input and target vectors, suitable for testing.
data TwoLayerTestData = 
  TLTD (ColumnVector Double) Layer Layer (ColumnVector Double)
    deriving Show

-- | Generate a layer with suitable input and target vectors, of the specified "size", 
-- | with arbitrary values.
-- | To see sample values (of "size" 4), in GHCi type: sample (sizedTwoLayerTestData 4)
sizedTwoLayerTestData :: Int -> Gen TwoLayerTestData
sizedTwoLayerTestData n = do
    l1 <- sizedArbLayer n
    l2 <- sizedArbLayer' (outputWidth l1) n
    x <- sizedArbColumnVector (inputWidth l1)
    t <- sizedArbColumnVector (outputWidth l2)
    return (TLTD x l1 l2 t)

instance Arbitrary TwoLayerTestData where
  -- | To see sample values, in GHCi type: sample arbTwoLayerTestData
  arbitrary = sized sizedTwoLayerTestData

-- | Training reduces error in a hidden layer
prop_trainingReducesHiddenLayerError :: TwoLayerTestData -> Property
prop_trainingReducesHiddenLayerError (TLTD x l1 l2 t)=
    -- (collect l) . -- uncomment to view test data
    (classifyRange "len x " n 0 25) .
    (classifyRange "len x " n 26 50) .
    (classifyRange "len x " n 51 75) .
    (classifyRange "len x " n 76 100) $
    errorAfter < errorBefore || errorAfter < 0.01
        where n = inputWidth l1
              pl0 = PropagatedSensorLayer{ pOut=x }
              pl1 = propagate pl0 l1
              pl2 = propagate pl1 l2
              bpl2 = backpropagateFinalLayer pl2 t
              bpl1 = backpropagate pl1 bpl2
              errorBefore = P.magnitude (t - pOut pl2) 
              l1New = update 0.00000001 bpl1 -- make sure we don't overshoot the mark 
              -- leave layer 2 alone, we're only interested in layer 1
              pl1New = propagate pl0 l1New
              pl2New = propagate pl1New l2
              errorAfter = P.magnitude (t - pOut pl2New)

classifyRange :: Testable a => String -> Int -> Int -> Int -> a -> Property
classifyRange s n n0 n1 = 
    classify (n >= n0 && n <= n1) (s ++ show n0 ++ ".." ++ show n1)

-- }}}

-- {{{ Common activation functions

data ActivationSpec = ActivationSpec
    {
      asF :: Double -> Double,
      asF' :: Double -> Double,
      desc :: String
    }

instance Show ActivationSpec where
    show = desc

identityAS = ActivationSpec
    {
      asF = id,
      asF' = const 1,
      desc = "identity"
    }

logisticSigmoidAS :: Double -> ActivationSpec
logisticSigmoidAS c = ActivationSpec
    {
        asF = logisticSigmoid c,
        asF' = logisticSigmoid' c,
        desc = "logistic sigmoid, c=" ++ show c
    }

arbitraryLogisticAS :: Gen ActivationSpec
arbitraryLogisticAS = do
    c <- choose(0,1) -- TODO Can it be > 1?
    return (logisticSigmoidAS c)

instance Arbitrary ActivationSpec where
--    arbitrary = oneof [ return identityAS, arbitraryLogisticAS ]
    arbitrary = return identityAS
        
logisticSigmoid :: (Field a, Floating a) => a -> a -> a
logisticSigmoid c a = 1 / (1 + exp((-c) * a))

logisticSigmoid' :: (Field a, Floating a) => a -> a -> a
logisticSigmoid' c a = (c * f a) * (1 - f a)
  where f = logisticSigmoid c
  

tanhAS :: ActivationSpec
tanhAS = ActivationSpec
    {
      asF = tanh,
      asF' = tanh',
      desc = "tanh"
    }

tanh' x = 1 - (tanh x)^2

-- }}}

