/**
 * Copyright Motphys Technology Co., Ltd. 2025, 2026
 * SPDX-License-Identifier: Apache-2.0
 */

function openVideoControls(video) {
    video.controls = true;
}

function isWeChatMobile() {
    const ua = navigator.userAgent.toLowerCase();
    return /micromessenger/.test(ua) && /android|iphone|ipad|ipod/.test(ua);
}

function setupWbtVideoDialog() {
    const videos = document.querySelectorAll("video.wbt-demo-video");
    if (!videos.length || isWeChatMobile()) {
        return;
    }

    const labels = document.documentElement.lang.toLowerCase().startsWith("zh")
        ? { close: "关闭放大视频", open: "点击放大视频", preview: "放大视频" }
        : { close: "Close enlarged video", open: "Enlarge video", preview: "Enlarged video" };
    const dialog = document.createElement("dialog");
    const closeButton = document.createElement("button");
    const enlargedVideo = document.createElement("video");
    let activeVideo = null;
    let resumeInlinePlayback = false;

    dialog.className = "video-lightbox-dialog";
    dialog.setAttribute("aria-label", labels.preview);
    closeButton.className = "video-lightbox-dialog-close";
    closeButton.type = "button";
    closeButton.setAttribute("aria-label", labels.close);
    closeButton.textContent = "×";
    enlargedVideo.controls = true;
    enlargedVideo.loop = true;
    enlargedVideo.playsInline = true;
    dialog.append(closeButton, enlargedVideo);
    document.body.append(dialog);

    function restoreInlineVideo() {
        if (!activeVideo) {
            return;
        }
        if (Number.isFinite(enlargedVideo.currentTime)) {
            activeVideo.currentTime = enlargedVideo.currentTime;
        }
        enlargedVideo.pause();
        enlargedVideo.onloadedmetadata = null;
        enlargedVideo.removeAttribute("src");
        enlargedVideo.load();
        if (resumeInlinePlayback) {
            activeVideo.play().catch(function () {});
        }
        activeVideo = null;
    }

    function openVideo(sourceVideo) {
        const source = sourceVideo.currentSrc || sourceVideo.querySelector("source")?.src;
        if (!source) {
            return;
        }
        if (typeof dialog.showModal !== "function") {
            if (typeof sourceVideo.requestFullscreen === "function") {
                sourceVideo.requestFullscreen();
            } else if (typeof sourceVideo.webkitEnterFullscreen === "function") {
                sourceVideo.webkitEnterFullscreen();
            }
            return;
        }

        activeVideo = sourceVideo;
        resumeInlinePlayback = !sourceVideo.paused;
        sourceVideo.pause();
        enlargedVideo.muted = sourceVideo.muted;
        enlargedVideo.poster = sourceVideo.poster;
        enlargedVideo.src = source;
        enlargedVideo.onloadedmetadata = function () {
            enlargedVideo.currentTime = sourceVideo.currentTime;
            enlargedVideo.play().catch(function () {});
        };
        dialog.showModal();
    }

    videos.forEach(function (video) {
        video.tabIndex = 0;
        video.setAttribute("role", "button");
        video.setAttribute("aria-label", labels.open);
        video.addEventListener("click", function () {
            openVideo(video);
        });
        video.addEventListener("keydown", function (event) {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                openVideo(video);
            }
        });
    });

    closeButton.addEventListener("click", function () {
        dialog.close();
    });
    dialog.addEventListener("click", function (event) {
        if (event.target === dialog) {
            dialog.close();
        }
    });
    dialog.addEventListener("close", restoreInlineVideo);
}

function setupTrainingCurveDialogs() {
    document.querySelectorAll("[data-wbt-curve-dialog], [data-training-curve-dialog]").forEach(function (trigger) {
        const dialogId = trigger.dataset.trainingCurveDialog || trigger.dataset.wbtCurveDialog;
        const dialog = document.getElementById(dialogId);
        if (!dialog || typeof dialog.showModal !== "function") {
            return;
        }
        trigger.addEventListener("click", function () {
            dialog.showModal();
        });
    });

    document.querySelectorAll(".wbt-curve-dialog, .training-curve-dialog").forEach(function (dialog) {
        dialog.querySelectorAll("[data-wbt-curve-close], [data-training-curve-close]").forEach(function (button) {
            button.addEventListener("click", function () {
                dialog.close();
            });
        });
        dialog.addEventListener("click", function (event) {
            if (event.target === dialog) {
                dialog.close();
            }
        });
    });
}

document.addEventListener("DOMContentLoaded", function () {
    if (isWeChatMobile()) {
        document.querySelectorAll("video").forEach(openVideoControls);
    }
    setupWbtVideoDialog();
    setupTrainingCurveDialogs();
});
