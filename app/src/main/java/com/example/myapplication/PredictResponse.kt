package com.example.myapplication

data class PredictionResponse(
    val recognized_object: String,
    val distance: Double
)