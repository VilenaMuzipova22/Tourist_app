package com.example.myapplication

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.navigation.findNavController
import androidx.navigation.ui.AppBarConfiguration
import androidx.navigation.ui.navigateUp
import androidx.navigation.ui.setupActionBarWithNavController
import com.example.myapplication.databinding.ActivityMainBinding
import com.google.android.material.snackbar.Snackbar
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var appBarConfiguration: AppBarConfiguration
    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private var imageCapture: ImageCapture? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Настройка Toolbar
        setSupportActionBar(binding.openSearchViewToolbar)

        // Настройка Navigation Component
        val navController = findNavController(R.id.nav_host_fragment_content_main)
        appBarConfiguration = AppBarConfiguration(navController.graph)
        setupActionBarWithNavController(navController, appBarConfiguration)

        // Обработка нажатия на FAB
        binding.fab.setOnClickListener { view ->
            Snackbar.make(view, "Replace with your own action", Snackbar.LENGTH_LONG)
                .setAction("Action", null)
                .setAnchorView(R.id.fab).show()
        }

        // Инициализация Executor для камеры
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Проверка разрешений и запуск камеры
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissions.launch(arrayOf(Manifest.permission.CAMERA))
        }

        // Настройка кнопки захвата изображения
        binding.captureButton.setOnClickListener {
            captureAndSaveImage()
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Настройка Preview
            val preview = Preview.Builder().build()
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            // Настройка ImageCapture
            imageCapture = ImageCapture.Builder().build()

            // Связываем Preview с PreviewView
            preview.setSurfaceProvider(binding.previewView.surfaceProvider)

            try {
                // Отвязываем все UseCase перед повторной привязкой
                cameraProvider.unbindAll()

                // Привязываем UseCase к жизненному циклу
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture
                )
            } catch (exc: Exception) {
                Log.e("CameraX", "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun captureAndSaveImage() {
        if (imageCapture == null) {
            Log.e("CameraX", "ImageCapture is null")
            return
        }

        val cacheDir = externalCacheDir
        if (cacheDir == null) {
            Log.e("CameraX", "Cache directory is null")
            return
        }

        val file = File(cacheDir, "captured_image.jpg")
        val outputOptions = ImageCapture.OutputFileOptions.Builder(file).build()

        imageCapture?.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    Log.d("CameraX", "Image saved: ${file.absolutePath}")
                    uploadImage(file)
                }

                override fun onError(exc: ImageCaptureException) {
                    Log.e("CameraX", "Photo capture failed: ${exc.message}", exc)
                }
            }
        )
    }

    private fun uploadImage(file: File) {
        Log.d("UploadImage", "File path: ${file.absolutePath}")
        Log.d("UploadImage", "File exists: ${file.exists()}")
        Log.d("UploadImage", "File size: ${file.length()} bytes")

        val imagePart = createImagePart(file)

        RetrofitClient.apiService.uploadImage(imagePart).enqueue(object : Callback<PredictionResponse> {
            override fun onResponse(call: Call<PredictionResponse>, response: Response<PredictionResponse>) {
                if (response.isSuccessful) {
                    val prediction = response.body()
                    Log.d("API", "Recognized object: ${prediction?.recognized_object}, Distance: ${prediction?.distance}")
                    runOnUiThread {
                        Snackbar.make(binding.root, "Recognized: ${prediction?.recognized_object}", Snackbar.LENGTH_LONG).show()
                    }
                } else {
                    Log.e("API", "Error code: ${response.code()}, Error body: ${response.errorBody()?.string()}")
                    runOnUiThread {
                        Snackbar.make(binding.root, "Server error: ${response.errorBody()?.string()}", Snackbar.LENGTH_LONG).show()
                    }
                }
            }

            override fun onFailure(call: Call<PredictionResponse>, t: Throwable) {
                Log.e("API", "Failed to upload image", t)
                runOnUiThread {
                    Snackbar.make(binding.root, "Network error: ${t.message}", Snackbar.LENGTH_LONG).show()
                }
            }
        })
    }

    private fun createImagePart(file: File): MultipartBody.Part {
        val mediaType = "image/jpeg".toMediaType() // Используем toMediaType
        val requestFile = file.asRequestBody(mediaType)
        return MultipartBody.Part.createFormData("file", file.name, requestFile)
    }

    private fun allPermissionsGranted() = arrayOf(Manifest.permission.CAMERA).all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private val requestPermissions = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        if (permissions[Manifest.permission.CAMERA] == true) {
            startCamera()
        } else {
            Snackbar.make(binding.root, "Camera permission is required", Snackbar.LENGTH_LONG).show()
        }
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.menu_main, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.action_settings -> true
            else -> super.onOptionsItemSelected(item)
        }
    }

    override fun onSupportNavigateUp(): Boolean {
        val navController = findNavController(R.id.nav_host_fragment_content_main)
        return navController.navigateUp(appBarConfiguration) || super.onSupportNavigateUp()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}