def evaluate_disp_metrics():
    
    # Select model type
    # model_type = ModelType.middlebury
    # model_type = ModelType.flyingthings
    model_type = ModelType.eth3d

    if model_type == ModelType.middlebury:
        model_path = "models/middlebury_d400.pb"
    elif model_type == ModelType.flyingthings:
        model_path = "models/flyingthings_finalpass_xl.pb"
    elif model_type == ModelType.eth3d:
        model_path = "models/eth3d.pb"

    results = {}
    dstype = "simdata"

    test_loader = fetch_dataloader()
    epe_list = []
    rmse_list = []
    bad_pixel_ratio_list_2 = []
    bad_pixel_ratio_list_1 = []

    hitnet_depth = HitNet(model_path, model_type)


    for i_batch, data_blob in enumerate(tqdm(test_loader)):
        
        image1, image2, right_img, depth1, depth2, flow_gt, z_flow_map = [data_item.cuda() for data_item in data_blob]

        image1_padded, image2_padded = prepare_images_and_depths(image1, image2)
        
        image1_np = image1.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image2_np = image2.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        disp = hitnet_depth(image1_np, image2_np)
        
        
  
        disp_gt = 800.0 * 0.12 / (depth1.squeeze(0).cpu().numpy()  )
        epe = np.mean(np.sqrt((disp - disp_gt) ** 2))
        rmse = np.sqrt(np.mean((disp - disp_gt) ** 2))
        bad_pixels_2 = (np.abs(disp - disp_gt) > 2).astype(np.float32).mean()
        bad_pixels_1 = (np.abs(disp - disp_gt) > 1).astype(np.float32).mean()


        epe_list.append(epe)
        rmse_list.append(rmse)
        bad_pixel_ratio_list_2.append(bad_pixels_2)
        bad_pixel_ratio_list_1.append(bad_pixels_1)

  

        disparity_colormap = draw_disparity(disp)
  
        output = np.vstack([image1_np.astype(np.uint8),disparity_colormap])
        output = cv2.resize(output, None, fx = 0.5, fy = 0.5)
  
        # cv2.imshow("Estimated depth", output)
        # cv2.waitKey(10)

    results['epe'] = np.mean(epe_list)
    results['rmse'] = np.mean(rmse_list)
    results['bad_pixel 2.0'] = np.mean(bad_pixel_ratio_list_2)
    results['bad_pixel 1.0'] = np.mean(bad_pixel_ratio_list_1)

    ic(results)
    return results