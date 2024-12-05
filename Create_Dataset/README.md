# Membuat Dataset

Dalam tahap membuat suatu program Object Detection ada beberapa hal yang harus disiapkan terlebih dahulu, salah satunya menyiapkan dataset yang akan digunakan. Ada beberapa cara untuk menyiapkan atau membuat dataset, dalam materi kali ini akan dijelaskan cara dalam membuat dataset menggunakan **Roboflow**.

# Roboflow

<p align="center">
    <img src='/contents/logoRoboflow.png' style="vertical-align:middle">
</p>

Roboflow merupakan salah satu platform berbasis web yang berfungsi untuk membuat dataset pada computer vision. Pada Roboflow sendiri memiliki beberapa fitur yang dapat mempermudah dalam proses pembuatan dataset sendir, seperti :

- `Manajemen Data`, mulai dari mengunggah gambar atau video, anotasi atau memberi label pada gambar sampai mengolah dataset.
- `Augmentasi Data`, untuk memperbanyak variasi dari gambar (memutar gambar, mengubah warna, dll) yang mana bertujuan supaya model lebih pintar dan akurat dalam mengenali berbagai jenis gambar.
- `Training Model`, pada Roboflow sudah disediakan model bawaan yang dapat langsung digunakan untuk melatih data yang sudah di buat.
- `Integrasi Mudah`, setelah dataset berhasil dibuat dan dilatih maka kita langsung bisa menggunakan nya dengan cara menarik API dari dataset tersebut.

Setelah mengetahui sekilas tentang `Roboflow`, maka kita langsung saja masuk ke praktik cara membuat dataset di Roboflow.

1. Langkah pertama, pastikan sudah memiliki kumpulan gambar atau video yang akan dijadikan dataset. `Sebisa mungkin bervariasi, jangan hanya gambar muka dari bagian depan saja`.

2. Langkah selanjutnya bagi teman-teman yang belum memiliki akun `Roboflow` bisa untuk mendaftar atau login terlebih dahulu. Boleh menggunakan akun Google atau Github.

<p align="center">
    <img src='/contents/signinRoboflow.png' style="vertical-align:middle">
</p>

3. Ketika sudah berhasil login, teman-teman bisa klik `New Projet` dipojok kanan atas.

<p align="center">
    <img src='/contents/addnewroboflow.png' style="vertical-align:middle">
</p>

4. Langkah selanjutnya, teman-teman membuat project baru. Isi kolom `Project Name` sesuai dengan project teman-teman. Lalu pada kolom `Annotating Group` isi dengan nama kelas dari gambarnya (Nama teman kelompok, hewan, dll). Dan pada bagian `Project Type` sesuaikan dengan kebutuhan teman-teman (Karena tugasnya object detection maka pilih yang itu). Kalau sudah klik `Create Public Project`.

<p align="center">
    <img src='/contents/createprojectrobo.png' style="vertical-align:middle">
</p>

5. Langkah yang kelima upload data gambar yang akan kalian jadikan dataset. Setelah itu klik `Save and Continue`.

<p align="center">
    <img src='/contents/uploaddatarobo.png' style="vertical-align:middle">
</p>

6. Pilih bagian `Manual Label`. Fungsinya agar teman-teman bisa melakukan anotasi secara manual, biar bisa disesuaikan juga mau bagian mana yang kalian anotasikan.

<p align="center">
    <img src='/contents/manuallabelrobo.png' style="vertical-align:middle">
</p>

7. Pada tampilan selanjutnya teman-teman bisa mengundang akun teman satu tim untuk melakukan labeling gambar secara bersamaan. Setelah itu klik `Assign`.

<p align="center">
    <img src='/contents/addteammatesrobo.png' style="vertical-align:middle">
</p>

8. Setelah itu klik bagian `Start Annotating`.

<p align="center">
    <img src='/contents/annotatingrobo.png' style="vertical-align:middle">
</p>

9. Untuk memulai anotasi gambar, teman-teman bisa klik bagian `Bounding Box`. Setelah itu langsung aja anotasi bagian muka kalian.

<p align="center">
    <img src='/contents/boundingboxrobo.png' style="vertical-align:middle">
</p>

10. Nah dibagian ini kalian sesuaikan bounding box nya sama muka kalian, jangan kegedean jangan kekecilan. Sebisa mungkin sesuaikan dengan mukanya. Nah kalau sudah dibagian pojok kiri atas nanti akan muncul kotak `Annotation Editor`, klik `Save` kalau muka dan nama kelas nya sudah sesuai. Contoh disini nama kelas nya `Junet`.

<p align="center">
    <img src='/contents/anotasimukarobo.png' style="vertical-align:middle">
</p>

11. Nah kalau belum sesuai atau mau kalian tambahkan nama teman kelompok kalian untuk kelas nya, kalian tinggal tulis nama kelas yang baru terus klik `Create Class` lalu `Save`. Seperti gambar dibawah. Kalau sudah di tambahkan kelas baru, nanti dia akan muncul list kelas-kelas nya jadi kalian tidak perlu menambahkan cara ini untuk nama kelas yang sama. Tinggal klik nama kelasnya dari list yang ditampilkan.

<p align="center">
    <img src='/contents/addnewclassrobo.png' style="vertical-align:middle">
</p>

12. Apabila sudah selesai melakukan anotasi, kalian klik panah di pojok kiri atas untuk kembali ke tampilan sebelumnya. Setelah itu klik `Add Image to Dataset` dipojok kanan atas.

<p align="center">
    <img src='/contents/addimagetodatasetrobo.png' style="vertical-align:middle">
</p>

13. Selanjutnya akan keluar tampilan pembagian dataset. Kalian sesuaikan saja dengan keinginan kalian. `Coba cari tau sendiri persentase untuk pembagian datanya`. Setelah itu klik `Add Image`.

<p align="center">
    <img src='/contents/splitdatarobo.png' style="vertical-align:middle">
</p>

14. Setelah berhasil atau selesai melakukan anotasi pada gambar yang kalian miliki, kita akan berpindah kebagian `Dataset` untuk melakukan Train Model. Caranya seperti gambar dibawah.

<p align="center">
    <img src='/contents/traindatasetrobo.png' style="vertical-align:middle">
</p>

15. Setelah itu akan muncul tampilan `Generate Dataset`. Nah teman-teman dapat mengatur sendiri proses yang akan dilakukan terhadap gambar yang sudah di anotasi sebelumnya, mulai dari pre-processing sampai ke augmentasinya. `Coba cari tau kira-kira proses yang bagus untuk membuat dataset wajah apa saja, mulai dari segi processing nya dan augmentasinya.` Dikembalikan kepada kelompok masing-masing, bebas sesuai dengan keinginan. Nah kalau sudah selesai klik `Create`.

<p align="center">
    <img src='/contents/addpreprocessingdatarobo.png' style="vertical-align:middle">
</p>

16. Sampai tahap ini, dataset yang teman-teman buat sudah dapat digunakan. Caranya gimana? tinggal klik `Download Dataset` dipojok kanan atas, lalu akan muncul tampilan download. Pada bagian ini teman-teman dapat menyesuaikan untuk cara mendownloadnya. Kalau ikut tutorial ini, kita langsung mendownload dataset lewat code nya. Untuk format bisa teman-teman sesuaikan lagi dengan yang teman-teman pakai. Setelah itu klik `Continue`.

<p align="center">
    <img src='/contents/downoaddatasetrobo.png' style="vertical-align:middle">
</p>

17. Nah ini terakhir banget. Kodenya bisa langsung teman-teman salin dan digunakan di kodingan teman-teman.

<p align="center">
    <img src='/contents/copyapirobo.png' style="vertical-align:middle">
</p>

Sekian tutorial singkat membuat dataset menggunakan `Roboflow` ini, semoga bisa dimengerti oleh teman-teman.

Teman-teman juga bisa melihat tutorial Roboflow melalui vidio di bawah ini.
[Tutor Roboflow](https://www.youtube.com/watch?v=NenFL5EgY_o)

**Terima Kasih**
