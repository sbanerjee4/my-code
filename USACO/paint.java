import java.util.*;
import java.io.*;
import java.lang.*;
public class paint
{
   static Rect[] a;
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("paint.in"));
      int n = in.nextInt();
      a = new Rect[n];
      int minx = Integer.MAX_VALUE, maxx = Integer.MIN_VALUE;
      for(int i = 0; i < n; i++)
      {
         int e = in.nextInt(), b = in.nextInt(), c = in.nextInt(), d = in.nextInt();
         a[i] = new Rect(e, b, c, d);
         minx = Math.min(minx, e);
         maxx = Math.max(maxx, c);
      }
      
      Arrays.sort(a);
      
      long area = 0;
      for(int i = minx; i <= maxx; i++)
      {
         ArrayList<Point> list = new ArrayList<>();
         for(int j = 0; j < n; j++)
         {
            if(a[j].x <= i && a[j].x2 >= i + 1)
            {
               list.add(new Point(a[j].y, -1));
               list.add(new Point(a[j].y2, 1));
            }
         }
         Collections.sort(list);
         if(list.size() > 0)
         {
            int strip = 0, open = 1, start = list.get(0).y;
            for(int j = 1; j < list.size(); j++)
            {
               if(open == 0) start = list.get(j).y;
               open += list.get(j).label;
               if(open == 0) strip += list.get(j).y - start;
            }
         
            area += strip;
         }
      }
      
      System.out.println(area);
   }
   
   static class Point implements Comparable<Point>
   {
      public int y;
      public int label;
      public Point(int b, int c)
      {
         y = b;
         label = c;
      }
      
      public int compareTo(Point a)
      {
         if(y == a.y) 
            return -1 * Integer.compare(label, a.label);
         return Integer.compare(y, a.y);
      }
   }
   
   static class Rect implements Comparable<Rect>
   {
      public int x;
      public int y;
      public int x2;
      public int y2;
      public Rect(int a, int b, int c, int d)
      {
         x = a;
         y = b;
         x2 = c;
         y2 = d;
      }
        
      public int compareTo(Rect a)
      {
         if(x == a.x) 
            return Integer.compare(y, a.y);
         return Integer.compare(x, a.x);
      }
   }
}