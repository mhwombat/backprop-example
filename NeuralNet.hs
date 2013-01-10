module NeuralNet where

import MatrixPlus

import Numeric.LinearAlgebra as N
import Test.QuickCheck

class NeuralNet n where
  evaluate
    :: n        -- ^ The neural net
    -> [Double] -- ^ The input pattern
    -> [Double] -- ^ The output pattern
  train
    :: n        -- ^ The neural net before training
    -> [Double] -- ^ The input pattern
    -> [Double] -- ^ The target pattern
    -> n        -- ^ The neural net after training

-- {{{ Useful things for testing

-- | Generate an arbitrary number between zero and one.
-- | To see sample values, in GHCi type: sample arbZeroToOne
arbZeroToOne :: Gen Double
arbZeroToOne = choose (0, 1)

-- | Generate a column vector of the specified length with arbitrary values.
-- | To see sample values (of length 4), in GHCi type: sample (sizedArbColumnVector 4)
sizedArbColumnVector :: Int -> Gen (ColumnVector Double)
sizedArbColumnVector n = do
    values <- vectorOf n arbZeroToOne
    return (listToColumnVector values)

-- | Generate a column vector where all values are zero
zeroColumnVector :: Int -> ColumnVector Double
zeroColumnVector n = listToColumnVector (replicate n 0)

-- | Generate a weight matrix of the specified dimensions with arbitrary values.
-- | To see sample values (of size 3x2), in GHCi type: sample (sizedArbWeightMatrix 3 2)
sizedArbWeightMatrix :: Int -> Int -> Gen (Matrix Double)
sizedArbWeightMatrix r c = do
    values <- vectorOf (r*c) arbZeroToOne
    return ((r N.><c) values)


-- }}}

