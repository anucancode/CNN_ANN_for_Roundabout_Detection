clear; close all; clc

% For getting the database file (.kml) use url: https://roundabouts.kittelson.com/Roundabouts/Search#
database = readgeotable('roundabout_export_4_30_2024.kml');
zoomlevel = 25;

for i = 1 : height(database)
    lat = database(i,1).Shape.Latitude;
    lon = database(i,1).Shape.Longitude;
    ofs = 0.0005;
    limLat = [lat-ofs, lat+ofs];
    LimLon = [lon-ofs, lon+ofs];

    [x1,y1,~]=latlon2local(limLat,LimLon,zeros(size(limLat)),[limLat(1),LimLon(1),0]);

    [ARange,~,~] = readBasemapImage('satellite',limLat,LimLon,zoomlevel);

    filename=strcat('pictures/',num2str(i),'I_',num2str(abs(x1(2))),'_',num2str(abs(y1(2))),'_','.png');
    imwrite(ARange,filename);

    clc;
    disp(i/height(database)*100);
end