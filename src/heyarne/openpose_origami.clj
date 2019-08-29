(ns heyarne.openpose-origami
  (:require [opencv4.dnn :as dnn]
            [opencv4.colors.rgb :as rgb]
            [opencv4.utils :as utils]
            [opencv4.core :as cv])
  (:import [org.opencv.core Point]))

;; this is basically a port of...
;; - https://github.com/chungbwc/Magicandlove/blob/master/ml20180806b/ml20180806b.pde
;; - https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/

(def cell 46) ;; <- NOTE: This is just taken blindly from Magicandlove
(def threshold 0.1)

#_(def bodyparts
  {::left-shoulder [1 2]
   ::right-shoulder [1 5]
   ::left-arm [2 3]
   ::left-forearm [3 4]
   ::right-arm [5 6]
   ::right-forearm [6 7]
   ::left-body [1 8]
   ::left-thigh [8 9]
   ::left-calf [9 10]
   ::right-body [1 11]
   ::right-thigh [11 12]
   ::right-cal [12 13]
   ::neck [1 0]
   ::left-nose [0 14]
   ::left-eye [14 16]
   ::right-nose [0 15]
   ::right-eye [15 17]})

(def net
  (dnn/read-net-from-caffe "resources/openpose_pose_coco.prototxt" "resources/pose_iter_440000.caffemodel"))

(def img (cv/imread "resources/david-foster-wallace-small.jpg" cv/IMREAD_COLOR))
(def scale-x (float (/ (.width img) cell)))
(def scale-y (float (/ (.height img) cell)))

(def blob
  (dnn/blob-from-image img
                       (/ 1.0 255)
                       (cv/new-size (.width img) (.height img))
                       (cv/new-scalar 0 0 0)
                       false false))

(.setInput net blob)
(time
 (def output (.. net forward (reshape 1 19)))) ;; <- 19 bodyparts

(defn point->vec
  "Converts an org.opencv.Point into something that's nicer to work with"
  [pt]
  [(.-x pt) (.-y pt)])

(defn points [output]
  (->>
   (range 18)
   ;; retrieve points as vectors from heatmap
   (keep (fn [i]
           (let [heatmap (.. output (row i) (reshape 1 cell))
                 min-max (cv/min-max-loc heatmap)]
             (.release heatmap)
             (when (> (.-maxVal min-max) threshold)
               (point->vec (.-maxLoc min-max))))))
   ;; rescale points to original size
   (map (fn [[x y]] [(* scale-x x) (* scale-y y)]))))

(def radius 5)
(def thickness 1)

(defn draw-circles! [mat points color radius thickness]
  (doseq [[x y] points]
    (cv/circle mat (Point. x y)
               radius color thickness))
  mat)

(defn -main [& args]
  (-> (cv/clone img)
      (draw-circles! (points output) rgb/greenyellow radius thickness)
      (utils/imshow)))

;; (-main)
