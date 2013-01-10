module MatrixPlus where

import Numeric.LinearAlgebra as N
import Test.QuickCheck

dummyMatrix :: (Field a) => Matrix a
dummyMatrix = (0 N.><0) []

mapMatrix :: (Field a)
  => (a -> a)
  -> Matrix a
  -> Matrix a
mapMatrix f x = (r N.><c) y
  where x' = toList (flatten x)
        y = map f x'
        r = rows x
        c = cols x

zipMatricesWith :: (Field a)
  => (a -> a -> a)
  -> Matrix a
  -> Matrix a
  -> Matrix a
zipMatricesWith f x y = (r N.><c) z
  where x' = toList (flatten x)
        y' = toList (flatten y)
        z = zipWith f x' y'
        r = rows x
        c = cols x

hadamardProduct :: (Field a)
  => Matrix a
  -> Matrix a
  -> Matrix a
hadamardProduct = zipMatricesWith (*)

average :: (Field a)
  => Matrix a
  -> a
average m = sum ms / fromIntegral (length ms)
  where ms = toList (flatten m)

magnitude :: (Field a, Floating a)
  => Matrix a
  -> a
magnitude x = 
  if cols x == 1
  then sqrt (sum xsxs)
  else error "not a column vector"
    where xs = toList (flatten x)
          xsxs = zipWith (*) xs xs

pseudoMagnitude :: (Field a, Floating a)
  => Matrix a
  -> a
pseudoMagnitude m = sqrt (sum msms)
    where ms = toList (flatten m)
          msms = zipWith (*) ms ms

-- | Inputs, outputs and targets are represented as column vectors instead of lists
type ColumnVector a = Matrix a

-- | Convert a list to a column vector
listToColumnVector :: (Ord a, Field a)
    -- | the list to convert
    => [a]
    -- | the resulting column vector
    -> ColumnVector a
listToColumnVector x = (len N.><1 ) x
    where len = length x

-- | Convert a column vector to a list
columnVectorToList :: (Ord a, Field a)
    -- | The column vector to convert
    => ColumnVector a
    -- | The resulting list
    -> [a]
columnVectorToList = toList . flatten

-- | Testable property:
-- | If we take a non-empty list, we should be able to convert it to a column vector
-- | and back, and get the result we started with.
-- | To test this property, in GHCi type: quickCheck prop_tocolumnVectorToListRoundtrippable
prop_tocolumnVectorToListRoundtrippable :: [Double] -> Property
prop_tocolumnVectorToListRoundtrippable x =
    (length x > 0) ==>
        -- collect x $ -- uncomment to view test data
        (columnVectorToList (listToColumnVector x) == x)

