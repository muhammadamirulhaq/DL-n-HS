module Main where

import Data.List (transpose)
import Control.Monad (when)
import Control.Exception (evaluate)
import Control.DeepSeq (deepseq)
import Text.Printf (printf)

(*.) x y = [ [sum $ zipWith (*) x y | x <- x] | y <- transpose y ] -- matmul
(+.) x y = [ [x+y | (x,y) <- zip x y] | (x,y) <- zip x y ] -- matadd

relu = max 0
sigmoid x = 1/(1 + exp (-x))
activation f x = [ [f x | x <- x] | x <- x ]

-- Linear Congruential Generator (LCG)
lcg :: Int -> [Int]
lcg seed = iterate (\x -> (a * x + c) `mod` m) seed
  where
    a = 1103515245
    c = 12345
    m = 2^31 - 1

randomGEN seed a b =
  map (\x->x`mod`(b-a+1) + a)
    $ lcg seed

randomF :: Int -> (Float, Float) -> [Float]
randomF seed (a,b) = map (\x -> fromIntegral x/100)
  $ randomGEN seed (round(a*100)) (round(b*100))

matrixFromList :: Int -> Int -> [a] -> [[a]]
matrixFromList n m = take n . map (take m) . iterate (drop m)

x = [[1,1],[1,0],[0,1],[0,0]]
yTrue = [0,1,1,0]

w1 = matrixFromList 2 3 (randomF 42 (-0.1,0.1))
b1 = matrixFromList 3 1 (randomF 41 (-0.1,0.1))
w2 = matrixFromList 3 1 (randomF 40 (-0.1,0.1))
b2 = matrixFromList 1 1 (randomF 39 (-0.1,0.1))

feedforward idx (w1, b1, w2, b2) = (hidden, output)
  where
    hidden = activation sigmoid ([x!!idx]*.w1 +. b1)
    output = activation sigmoid (transpose (hidden)*.w2 +. b2)

backpropagation epoch (w1, b1, w2, b2) lr = (w1', b1', w2', b2')
  where
    idx = epoch `mod` 4
    (hidden, output) = feedforward idx (w1, b1, w2, b2)
    yPred = head $ head output
    errGrad = ((yTrue!!idx)-yPred)*yPred*(1-yPred)
    deltaW2 3 = lr*errGrad*head (head b2)
    deltaW2 j = lr*errGrad*head (hidden!!j)
    errNet j = errGrad*head (w2!!j)
    errGrad2 j = errNet j*head (hidden!!j)*(1-head (hidden!!j))
    deltaW1 j 2 = lr*errGrad2 j
    deltaW1 j i = lr*errGrad2 j*head [x!!idx]!!i
    w2New 3 = head (head b2) + deltaW2 3
    w2New j = head (w2!!j) + deltaW2 j
    w1New j 2 = head (b1!!j) + deltaW1 j 2
    w1New j i = w1!!i!!j + deltaW1 j i
    w1' = [[w1New 0 0, w1New 1 0, w1New 2 0], [w1New 0 1, w1New 1 1, w1New 2 1]]
    b1' = [[w1New 0 2], [w1New 1 2], [w1New 2 2]]
    w2' = [[w2New 0], [w2New 1], [w2New 2]]
    b2' = [[w2New 3]]

train i numEpochs initial lr =
  when (i`mod`10000 == 0) (putStrLn $ "Epoch " ++ show i ++ "/" ++ show numEpochs) >>
  let weights = backpropagation i initial lr
  in
    strictEval weights >>= (\ws ->
      if i == numEpochs-1 then pure ws
      else train (i+1) numEpochs ws lr
      )

  where
    strictEval x = do
      x' <- evaluate x
      let xEvaluated = x' `deepseq` x'
      pure xEvaluated


main = do
  weights <- train 0 100000 (w1, b1, w2, b2) 0.4
  let (_,res1) = feedforward 0 weights
      (_,res2) = feedforward 1 weights
      (_,res3) = feedforward 2 weights
      (_,res4) = feedforward 3 weights

  putStrLn "Training Completed."
  printf "1 xor 1 -> %.4f\n" (res1!!0!!0)
  printf "1 xor 0 -> %.4f\n" (res2!!0!!0)
  printf "0 xor 1 -> %.4f\n" (res3!!0!!0)
  printf "0 xor 0 -> %.4f\n" (res4!!0!!0)