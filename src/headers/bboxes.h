#ifndef BBOXES_H
#define BBOXES_H

class BBoxes{
public:
    struct BBox2D{
        int cls;
        int w;
        int h;
        int lx; 
        int ly;
        int rx; 
        int ry;
        int cx;
        int cy;
    };
    struct BBox3D{
        int cls;
        float cx;
        float cy;
        float cz;
        float dx; //width
        float dy; //length
        float dz; //height
    };

    struct Corner3D{
        float x;
        float y;
        float z;
    };

    struct Corners3D{
    /*
        6 -------- 7
       /|         /|
      5 -------- 4 .
      | |        | |
      . 2 -------- 3
      |/         |/
      1 -------- 0
    */
        Corner3D cor0;
        Corner3D cor1;   
        Corner3D cor2; 
        Corner3D cor3; 
        Corner3D cor4; 
        Corner3D cor5; 
        Corner3D cor6; 
        Corner3D cor7; 
    };

    BBoxes::Corners3D calcCorners(BBoxes::BBox3D bbox){
        /*
            (-1,-1,1) -------- (-1,1,1)
            /|               /|
        (1,-1,1) -------- (1,1,1) .
            | |              | |
        . (-1,-1,-1) -------- (-1,1,-1)
            |/               |/
        (1,-1,-1) -------- ( 1,1,-1)
        */
        BBoxes::Corners3D corners;
        corners.cor0.x = bbox.cx + bbox.dx/2; corners.cor0.y = bbox.cy + bbox.dy/2;  corners.cor0.z = bbox.cz - bbox.dz/2; 
        corners.cor1.x = bbox.cx + bbox.dx/2; corners.cor1.y = bbox.cy - bbox.dy/2;  corners.cor1.z = bbox.cz - bbox.dz/2; 
        corners.cor2.x = bbox.cx - bbox.dx/2; corners.cor2.y = bbox.cy - bbox.dy/2;  corners.cor2.z = bbox.cz - bbox.dz/2; 
        corners.cor3.x = bbox.cx - bbox.dx/2; corners.cor3.y = bbox.cy + bbox.dy/2;  corners.cor3.z = bbox.cz - bbox.dz/2; 
        corners.cor4.x = bbox.cx + bbox.dx/2; corners.cor4.y = bbox.cy + bbox.dy/2;  corners.cor4.z = bbox.cz + bbox.dz/2; 
        corners.cor5.x = bbox.cx + bbox.dx/2; corners.cor5.y = bbox.cy - bbox.dy/2;  corners.cor5.z = bbox.cz + bbox.dz/2; 
        corners.cor6.x = bbox.cx - bbox.dx/2; corners.cor6.y = bbox.cy - bbox.dy/2;  corners.cor6.z = bbox.cz + bbox.dz/2; 
        corners.cor7.x = bbox.cx - bbox.dx/2; corners.cor7.y = bbox.cy + bbox.dy/2;  corners.cor7.z = bbox.cz + bbox.dz/2; 
        return corners;
    }
};

#endif //BBOXES_H