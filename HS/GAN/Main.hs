module Main where

import Data.List (transpose)
import Control.Monad (when, forM_)
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

x = [[1.0,0.0],[0.94,0.05],[0.91,0.09],[0.92,0.08],[0.99,0.07]]
noise seed = take 2 $ randomF seed (-0.5,0.5)

w1 = matrixFromList 2 3 (randomF 42 (-0.5,0.5))
b1 = matrixFromList 3 1 (randomF 41 (-0.5,0.5))
w2 = matrixFromList 3 2 (randomF 40 (-0.5,0.5))
b2 = matrixFromList 2 1 (randomF 39 (-0.5,0.5))

w1_d = matrixFromList 2 3 (randomF 38 (-0.5,0.5))
b1_d = matrixFromList 3 1 (randomF 37 (-0.5,0.5))
w2_d = matrixFromList 3 1 (randomF 36 (-0.5,0.5))
b2_d = matrixFromList 1 1 (randomF 35 (-0.5,0.5))

generator noise (w1, b1, w2, b2) (w1_d,b1_d,w2_d,b2_d) = (hidden_g, output_g, hidden_d, output_d)
  where
    hidden_g = activation sigmoid ([noise]*.w1 +. b1)
    output_g = activation sigmoid (transpose hidden_g*.w2 +. b2)
    hidden_d = activation sigmoid (transpose output_g*.w1_d +. b1_d)
    output_d = activation sigmoid (transpose hidden_d*.w2_d +. b2_d)

discriminator epoch ((w1, b1, w2, b2),(w1_d,b1_d,w2_d,b2_d)) lr = ((w1', b1', w2', b2'),(w1_d_real',b1_d_real',w2_d_real',b2_d_real'))
  where
    noise_ = noise (43+epoch)
    idx = epoch `mod` 5
    (hidden_g, output_g, hidden_d, output_d) = generator noise_ (w1, b1, w2, b2) (w1_d,b1_d,w2_d,b2_d)
    yPred = head $ head output_d
    errGrad = (0-yPred)*yPred*(1-yPred)
    deltaW2 3 = lr*errGrad*head (head b2_d)
    deltaW2 j = lr*errGrad*head (hidden_d!!j)
    errNet j = errGrad*head (w2_d!!j)
    errGrad2 j = errNet j*head (hidden_d!!j)*(1-head (hidden_d!!j))
    deltaW1 j 2 = lr*errGrad2 j
    deltaW1 j i = lr*errGrad2 j*head [noise_]!!i
    w2New 3 = head (head b2_d) + deltaW2 3
    w2New j = head (w2_d!!j) + deltaW2 j
    w1New j 2 = head (b1_d!!j) + deltaW1 j 2
    w1New j i = w1_d!!i!!j + deltaW1 j i
    w1_d' = [[w1New 0 0, w1New 1 0, w1New 2 0], [w1New 0 1, w1New 1 1, w1New 2 1]]
    b1_d' = [[w1New 0 2], [w1New 1 2], [w1New 2 2]]
    w2_d' = [[w2New 0], [w2New 1], [w2New 2]]
    b2_d' = [[w2New 3]]
   
    hidden_d_real = activation sigmoid ([x!!idx]*.w1_d' +. b1_d')
    output_d_real = activation sigmoid (transpose hidden_d_real*.w2_d' +. b2_d')
    yPred_real = head $ head output_d_real
    errGrad_real = (1-yPred_real)*yPred_real*(1-yPred_real)
    deltaW2_real 3 = lr*errGrad_real*head (head b2_d')
    deltaW2_real j = lr*errGrad_real*head (hidden_d_real!!j)
    errNet_real j = errGrad_real*head (w2_d'!!j)
    errGrad2_real j = errNet_real j*head (hidden_d_real!!j)*(1-head (hidden_d_real!!j))
    deltaW1_real j 2 = lr*errGrad2_real j
    deltaW1_real j i = lr*errGrad2_real j*head [x!!idx]!!i
    w2New_real 3 = head (head b2_d) + deltaW2_real 3
    w2New_real j = head (w2_d!!j) + deltaW2_real j
    w1New_real j 2 = head (b1_d!!j) + deltaW1_real j 2
    w1New_real j i = w1_d!!i!!j + deltaW1_real j i
    w1_d_real' = [[w1New_real 0 0, w1New_real 1 0, w1New_real 2 0], [w1New_real 0 1, w1New_real 1 1, w1New_real 2 1]]
    b1_d_real' = [[w1New_real 0 2], [w1New_real 1 2], [w1New_real 2 2]]
    w2_d_real' = [[w2New_real 0], [w2New_real 1], [w2New_real 2]]
    b2_d_real' = [[w2New_real 3]]
   
    errNet_g j = errGrad2_real 0*(w1_d'!!j!!0) + errGrad2_real 1*(w1_d'!!j!!1) + errGrad2_real 2*(w1_d'!!j!!2)
    errGrad_g j = errNet_g j*head (output_g!!j)*(1-head (output_g!!j))
    deltaW2_g j 3 = lr*errGrad_g j
    deltaW2_g j i = lr*errGrad_g j*head (hidden_g!!i)
    errNet2_g j = errGrad_g 0*(w2!!j!!0) + errGrad_g 1*(w2!!j!!1)
    errGrad2_g j = errNet2_g j*head (hidden_g!!j)*(1-head (hidden_g!!j))
    deltaW1_g j 2 = lr*errGrad2_g j
    deltaW1_g j i =  lr*errGrad2_g j*(noise_!!i)
    w2New_g j 3 = head (b2!!j) + deltaW2_g j 3
    w2New_g j i = (w2!!i!!j) + deltaW2_g j i
    w1New_g j 2 = head (b1!!j) + deltaW1_g j 2
    w1New_g j i = (w1!!i!!j) + deltaW1_g j i
    w1' = [[w1New_g 0 0,w1New_g 1 0, w1New_g 2 0],[w1New_g 0 1,w1New_g 1 1, w1New_g 2 1]]
    b1' = [[w1New_g 0 2],[w1New_g 1 2], [w1New_g 2 2]]
    w2' = [[w2New_g 0 0,w2New_g 1 0],[w2New_g 0 1,w2New_g 1 1],[w2New_g 0 2, w2New_g 1 2]]
    b2' = [[w2New_g 0 3],[w2New_g 1 3]]

train i numEpochs initial lr =
  when (i`mod`10000 == 0) (putStrLn $ "Epoch " ++ show i ++ "/" ++ show numEpochs) >>
  let weights = discriminator i initial lr
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
  (weights1, weights2) <- train 0 100000 ((w1, b1, w2, b2),(w1_d,b1_d,w2_d,b2_d)) 0.5
  putStrLn "Training Completed."
  forM_ [0..4] (\i -> do
    let (_,res,_,_) = generator (noise (43+i)) weights1 weights2
    putStrLn $ "Generate "++show (i+1)++" -> " ++ show res
    )
