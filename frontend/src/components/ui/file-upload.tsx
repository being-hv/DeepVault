"use client";

/**
 * USE CASE: Interactive Drag-and-Drop Uploader
 * This component provides an interface supporting both clicking to select local files 
 * and dragging-and-dropping image files from the system.
 * It manages drag/drop window hover states and coordinates loading spinner states.
 */

import React, { useCallback, useState } from "react";

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  isLoading: boolean;
}

export function FileUpload({ onFileSelect, isLoading }: FileUploadProps) {
  // tracks whether the user is currently hovering a file drag over the block area
  const [isDragging, setIsDragging] = useState(false);

  // Monitors drag entering or hovering the area to update styling states
  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragging(true);
    } else if (e.type === "dragleave") {
      setIsDragging(false);
    }
  }, []);

  // Monitors drag release/drop to ingest the file data
  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        const file = e.dataTransfer.files[0];
        // Ensure the dropped file is actually an image format
        if (file.type.startsWith("image/")) {
          onFileSelect(file);
        } else {
          alert("Please upload an image file.");
        }
      }
    },
    [onFileSelect]
  );

  // Ingests manual clicks/selection via browser file navigator dialog
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0]);
    }
  };

  return (
    <div
      className={`relative w-full p-8 border-2 border-dashed rounded-xl flex flex-col items-center justify-center transition-colors ${
        isDragging
          ? "border-blue-500 bg-blue-500/10"
          : "border-gray-600 bg-gray-800/50 hover:bg-gray-800"
      }`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
    >
      {/* Hidden browser input element triggered via the wrapper click wrapper */}
      <input
        type="file"
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        onChange={handleChange}
        accept="image/*"
        disabled={isLoading}
      />
      
      <div className="text-center pointer-events-none">
        {/* Upload Icon */}
        <svg
          suppressHydrationWarning={true}
          className="mx-auto h-12 w-12 text-gray-400 mb-4"
          stroke="currentColor"
          fill="none"
          viewBox="0 0 48 48"
          aria-hidden="true"
        >
          <path
            d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <p className="mt-1 text-sm text-gray-300">
          <span className="font-medium text-blue-400">Click to upload</span> or drag and drop
        </p>
        <p className="mt-1 text-xs text-gray-500">PNG, JPG, GIF up to 10MB</p>
      </div>

      {/* Full-bleed glassmorphism overlay spinning loader during server processing */}
      {isLoading && (
        <div className="absolute inset-0 bg-gray-900/50 flex items-center justify-center rounded-xl backdrop-blur-sm">
          <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-500"></div>
        </div>
      )}
    </div>
  );
}
